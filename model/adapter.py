import math
import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import SimpleAdapter, SimpleProj, MultiScaleFusion


class SpatialHighPass2D(nn.Module):
    """
    Lightweight image-only 2D spatial high-pass branch.
    Input:  patch tokens in (B, N, C)
    Output: residual-enhanced patch tokens in (B, N, C)

    Notes:
      - Only valid for patch tokens with a square spatial layout.
      - Uses a fixed depthwise Laplacian filter + learnable residual scale.
      - This module is NOT used in the text branch.
    """
    def __init__(self, channels: int, init_alpha: float = 0.05):
        super().__init__()
        lap = torch.tensor(
            [[0.0, -1.0, 0.0],
             [-1.0, 4.0, -1.0],
             [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer("kernel", lap.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.alpha_2d = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        if x.dim() != 3:
            return x

        B, N, C = x.shape
        H = W = int(N ** 0.5)
        if H * W != N:
            # non-square token length: keep safe fallback
            return x

        feat = x.transpose(1, 2).reshape(B, C, H, W)  # (B,C,H,W)
        hp = F.conv2d(feat, self.kernel.to(feat.dtype), padding=1, groups=C)
        feat = feat + self.alpha_2d * hp
        out = feat.reshape(B, C, N).transpose(1, 2)   # (B,N,C)
        return out


class ScoreCalibrator(nn.Module):
    """V28 lightweight dense-to-image score calibrator.

    It predicts a dense-score weight alpha in [alpha_min, alpha_max] from
    source-trained score statistics.  The module is intentionally tiny and
    only affects image-level scores when --use_score_calibrator is enabled.
    Pixel maps and DINOv3 residual fusion remain unchanged.
    """
    def __init__(self, in_dim: int = 9, hidden_dim: int = 32, alpha_min: float = 0.85, alpha_max: float = 1.0):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.alpha_max < self.alpha_min:
            lo, hi = self.alpha_max, self.alpha_min
        else:
            lo, hi = self.alpha_min, self.alpha_max
        alpha = lo + (hi - lo) * torch.sigmoid(self.net(features)).squeeze(-1)
        return alpha


class AdaptedCLIP(nn.Module):
    def __init__(
        self,
        clip_model,
        text_adapt_weight: float = 0.1,
        image_adapt_weight: float = 0.1,
        text_adapt_until: int = 3,
        image_adapt_until: int = 6,
        levels: list = [6, 12, 18, 24],
        relu: bool = True,
        enable_text_adapter: bool = True,
        enable_image_adapter: bool = True,
        use_dinov3: bool = False,
        dino_model_name: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        dino_fusion_alpha: float = 0.05,
        dino_img_size: int = None,
        dino_norm: str = "lvd",
        dino_local_files_only: bool = False,
        dino_residual_gate_mode: str = "none",
        dino_residual_gate_center: float = 0.0,
        dino_residual_gate_tau: float = 0.25,
        dino_residual_gate_min: float = 0.0,
        use_score_calibrator: bool = False,
        score_calib_alpha_min: float = 0.85,
        score_calib_alpha_max: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.text_adapt_until = text_adapt_until
        self.image_adapt_until = image_adapt_until
        self.levels = levels
        self.enable_text_adapter = enable_text_adapter
        self.enable_image_adapter = enable_image_adapter
        self.use_score_calibrator = bool(use_score_calibrator)
        self.score_calib_alpha_min = float(score_calib_alpha_min)
        self.score_calib_alpha_max = float(score_calib_alpha_max)

        # -------------------------------------------------
        # Optional frozen DINOv3 image branch.
        # Default is disabled, so original ICMR behaviour is unchanged.
        # -------------------------------------------------
        self.use_dinov3 = bool(use_dinov3)
        self.dino_model_name = dino_model_name
        self.dino_fusion_alpha = float(dino_fusion_alpha)
        self.dino_img_size = int(dino_img_size) if dino_img_size is not None else None
        self.dino_norm = str(dino_norm).lower()
        self.dino_residual_gate_mode = str(dino_residual_gate_mode).lower()
        self.dino_residual_gate_center = float(dino_residual_gate_center)
        self.dino_residual_gate_tau = max(float(dino_residual_gate_tau), 1e-6)
        self.dino_residual_gate_min = min(max(float(dino_residual_gate_min), 0.0), 1.0)
        self.dinov3_model = None
        self.dinov3_patch_size = 16
        dinov3_hidden_size = 1024

        self.register_buffer(
            "_clip_mean",
            torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_std",
            torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_dino_lvd_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_dino_lvd_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_dino_sat_mean",
            torch.tensor((0.430, 0.411, 0.296), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_dino_sat_std",
            torch.tensor((0.213, 0.156, 0.143), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if self.use_dinov3:
            try:
                from transformers import AutoModel
            except Exception as exc:
                raise ImportError(
                    "--use_dinov3 requires Hugging Face transformers. "
                    "Install a DINOv3-capable transformers version, e.g. transformers>=4.56.0."
                ) from exc

            self.dinov3_model = AutoModel.from_pretrained(
                self.dino_model_name,
                local_files_only=bool(dino_local_files_only),
            )
            self.dinov3_model.eval()
            self.dinov3_model.requires_grad_(False)
            dinov3_cfg = getattr(self.dinov3_model, "config", None)
            dinov3_hidden_size = int(getattr(dinov3_cfg, "hidden_size", dinov3_hidden_size))
            self.dinov3_patch_size = int(getattr(dinov3_cfg, "patch_size", 16))

        # -------------------------------------------------
        # 1) image / text adapter modules
        #    keep existing 1D SimpleAdapter unchanged
        #    add image-only 2D spatial high-pass branch
        # -------------------------------------------------
        layer_adapters = nn.ModuleList(
            [SimpleAdapter(1024, 1024) for _ in range(image_adapt_until)]
        )
        seg_proj = nn.ModuleList(
            [SimpleProj(1024, 768, relu) for _ in range(len(levels))]
        )
        det_proj = SimpleProj(1024, 768, relu)
        spatial_hp = nn.ModuleList(
            [SpatialHighPass2D(1024, init_alpha=0.05) for _ in range(len(levels))]
        )

        image_adapter_dict = {
            "layer_adapters": layer_adapters,
            "seg_proj": seg_proj,
            "det_proj": det_proj,
            "spatial_hp": spatial_hp,
        }
        if self.use_dinov3:
            # DINOv3 dense tokens are frozen, then projected into CLIP-ViT-L/14 token space.
            image_adapter_dict["dinov3_proj"] = SimpleProj(dinov3_hidden_size, 1024, relu=False)

        self.image_adapter = nn.ModuleDict(image_adapter_dict)

        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
            + [SimpleProj(768, 768, relu=True)]
        )

        # -------------------------------------------------
        # 2) per-layer learnable gates
        # -------------------------------------------------
        init_i = torch.logit(torch.tensor(image_adapt_weight))
        init_t = torch.logit(torch.tensor(text_adapt_weight))

        self.image_layer_gates = nn.Parameter(init_i.repeat(image_adapt_until))
        self.text_layer_gates = nn.Parameter(init_t.repeat(text_adapt_until))

        # -------------------------------------------------
        # 3) learnable multi-scale fusion
        # -------------------------------------------------
        self.ms_fusion = MultiScaleFusion(num_scales=len(self.levels))

        # V28: optional source-trained dense-to-image score calibrator.
        # Disabled by default so old checkpoints / inference stay compatible.
        self.score_calibrator = ScoreCalibrator(
            in_dim=9,
            hidden_dim=32,
            alpha_min=self.score_calib_alpha_min,
            alpha_max=self.score_calib_alpha_max,
        ) if self.use_score_calibrator else None

        self._init_weights_()

    def _init_weights_(self):
        for p in self.image_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if getattr(self, "score_calibrator", None) is not None:
            for p in self.score_calibrator.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def _preprocess_for_dinov3(self, image: torch.Tensor) -> torch.Tensor:
        # Dataset tensors are CLIP-normalized. Convert back to [0,1], then use DINOv3 normalization.
        x = image.float() * self._clip_std + self._clip_mean
        x = x.clamp(0.0, 1.0)

        if self.dino_img_size is not None and int(self.dino_img_size) > 0:
            x = F.interpolate(
                x,
                size=(int(self.dino_img_size), int(self.dino_img_size)),
                mode="bilinear",
                align_corners=False,
            )

        if self.dino_norm == "sat":
            mean, std = self._dino_sat_mean, self._dino_sat_std
        elif self.dino_norm == "none":
            return x
        else:
            mean, std = self._dino_lvd_mean, self._dino_lvd_std
        return (x - mean) / std

    def _extract_dinov3_tokens(self, image: torch.Tensor):
        if (not self.use_dinov3) or self.dinov3_model is None:
            return None, None

        pixel_values = self._preprocess_for_dinov3(image)
        B, _, H, W = pixel_values.shape
        grid_h = max(1, H // int(self.dinov3_patch_size))
        grid_w = max(1, W // int(self.dinov3_patch_size))
        expected_patches = grid_h * grid_w

        self.dinov3_model.eval()
        with torch.no_grad():
            outputs = self.dinov3_model(pixel_values=pixel_values, return_dict=True)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden = outputs[0]

        # DINOv3 ViT outputs can contain CLS/register tokens before patch tokens.
        # Keep the final patch-grid tokens for dense fusion.
        if hidden.shape[1] >= expected_patches:
            patch_tokens = hidden[:, -expected_patches:, :]
            return patch_tokens, (grid_h, grid_w)

        # Safe fallback if a backend returns patch tokens only with a non-standard grid.
        n = hidden.shape[1]
        hw = int(math.sqrt(n))
        if hw * hw == n:
            return hidden, (hw, hw)
        return None, None

    @staticmethod
    def _resize_patch_tokens(tokens: torch.Tensor, source_hw, target_n: int) -> torch.Tensor:
        if tokens is None or source_hw is None:
            return None
        target_hw = int(math.sqrt(int(target_n)))
        if target_hw * target_hw != int(target_n):
            return None
        src_h, src_w = int(source_hw[0]), int(source_hw[1])
        B, N, C = tokens.shape
        if src_h * src_w != N:
            src_hw = int(math.sqrt(N))
            if src_hw * src_hw != N:
                return None
            src_h = src_w = src_hw
        feat = tokens.transpose(1, 2).reshape(B, C, src_h, src_w)
        feat = F.interpolate(feat, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
        return feat.flatten(2).transpose(1, 2)

    def _compute_dino_residual_gate(self, clip_tokens: torch.Tensor, dino_projected: torch.Tensor) -> torch.Tensor:
        """Label-free reliability gate for the DINOv3 residual.

        Returns a gate in [dino_residual_gate_min, 1] with shape (B,N,1).
        Default mode ``none`` exactly recovers the original fixed residual fusion:
            t + alpha_DINO * dino_projected.

        ``agreement`` uses dense CLIP--DINO cosine agreement as reliability:
            r = sigmoid((cos(t, dino) - center) / tau).
        This is source/target-label free and can be used at test time with old checkpoints.
        """
        mode = getattr(self, "dino_residual_gate_mode", "none")
        if mode in ("none", "off", "fixed", "0"):
            return torch.ones(
                clip_tokens.shape[0], clip_tokens.shape[1], 1,
                device=clip_tokens.device, dtype=clip_tokens.dtype,
            )

        tau = max(float(getattr(self, "dino_residual_gate_tau", 0.25)), 1e-6)
        center = float(getattr(self, "dino_residual_gate_center", 0.0))
        gate_min = min(max(float(getattr(self, "dino_residual_gate_min", 0.0)), 0.0), 1.0)

        if mode in ("agreement", "clip_dino_agreement", "cosine"):
            clip_dir = F.normalize(clip_tokens.detach(), dim=-1)
            dino_dir = F.normalize(dino_projected.detach(), dim=-1)
            reliability = (clip_dir * dino_dir).sum(dim=-1, keepdim=True)
            gate = torch.sigmoid((reliability - center) / tau)
        elif mode in ("token_norm", "clip_norm"):
            # Fallback reliability when agreement is not desired: emphasize tokens whose
            # CLIP feature norm is above the image-level token average.
            norm = clip_tokens.detach().norm(dim=-1, keepdim=True)
            norm = (norm - norm.mean(dim=1, keepdim=True)) / (norm.std(dim=1, keepdim=True) + 1e-6)
            gate = torch.sigmoid((norm - center) / tau)
        else:
            raise ValueError(
                f"Unknown dino_residual_gate_mode={mode}. "
                "Use none, agreement, or token_norm."
            )

        if gate_min > 0.0:
            gate = gate_min + (1.0 - gate_min) * gate
        return gate.to(device=clip_tokens.device, dtype=clip_tokens.dtype)

    def _fuse_dinov3_into_tokens(self, tokens, image: torch.Tensor):
        if (not self.use_dinov3) or ("dinov3_proj" not in self.image_adapter):
            return tokens
        if self.dino_fusion_alpha == 0.0:
            return tokens

        dino_tokens, source_hw = self._extract_dinov3_tokens(image)
        if dino_tokens is None:
            return tokens

        fused_tokens = []
        for t in tokens:
            dino_resized = self._resize_patch_tokens(dino_tokens, source_hw, t.shape[1])
            if dino_resized is None:
                fused_tokens.append(t)
                continue
            dino_resized = dino_resized.to(device=t.device, dtype=t.dtype)
            dino_projected = self.image_adapter["dinov3_proj"](dino_resized)
            dino_projected = (
                dino_projected
                * t.norm(dim=-1, keepdim=True)
                / (dino_projected.norm(dim=-1, keepdim=True) + 1e-6)
            )
            gate = self._compute_dino_residual_gate(t, dino_projected)
            fused_tokens.append(t + float(self.dino_fusion_alpha) * gate * dino_projected)
        return fused_tokens


    @staticmethod
    def _dense_score_components_torch(pixel_map: torch.Tensor, topk_ratio: float = 0.01):
        """Return differentiable dense top-k score and detached reliability features."""
        if pixel_map.dim() == 4:
            flat = pixel_map.reshape(pixel_map.shape[0], -1)
        elif pixel_map.dim() == 3:
            flat = pixel_map.reshape(pixel_map.shape[0], -1)
        else:
            flat = pixel_map.reshape(pixel_map.shape[0], -1)
        n_pix = flat.shape[1]
        k = max(1, int(float(topk_ratio) * int(n_pix)))
        k = min(k, int(n_pix))
        topk_vals = torch.topk(flat, k=k, dim=1).values
        dense = topk_vals.mean(dim=1)
        mean = flat.mean(dim=1)
        std = flat.std(dim=1).clamp_min(1e-6)
        maxv = flat.max(dim=1).values
        med = flat.detach().median(dim=1).values
        mad = (flat.detach() - med[:, None]).abs().median(dim=1).values.clamp_min(1e-6)
        concentration = ((dense.detach() - med) / mad).clamp(-20.0, 20.0)
        high_thr = med + 2.0 * mad
        area_ratio = (flat.detach() > high_thr[:, None]).float().mean(dim=1)
        return dense, mean, std, maxv, concentration, area_ratio

    def calibrated_image_score(self, pixel_map: torch.Tensor, cls_score: torch.Tensor, topk_ratio: float = 0.01):
        """V28 image-score calibration.

        Args:
            pixel_map: (B,H,W) or (B,1,H,W), raw/suppressed anomaly evidence.
            cls_score: (B,), CLIP-style P(abnormal) or anomaly logit score.
        Returns:
            final_score, alpha, dense_score, feature_matrix
        """
        dense, mean, std, maxv, concentration, area_ratio = self._dense_score_components_torch(pixel_map, topk_ratio)
        cls_score = cls_score.reshape(-1).to(dense.dtype)
        abs_diff = (dense - cls_score).abs()
        feats = torch.stack([
            dense.detach(),
            cls_score.detach(),
            abs_diff.detach(),
            mean.detach(),
            std.detach(),
            maxv.detach(),
            concentration.detach(),
            area_ratio.detach(),
            torch.ones_like(dense.detach()) * float(topk_ratio),
        ], dim=1)
        if getattr(self, "score_calibrator", None) is None:
            alpha = torch.ones_like(dense) * float(getattr(self, "score_calib_alpha_max", 1.0))
        else:
            alpha = self.score_calibrator(feats)
        final_score = alpha * dense + (1.0 - alpha) * cls_score
        return final_score, alpha, dense, feats

    def forward_original(self, x, modality="visual"):
        if modality == "visual":
            cls_features, patch_features = self.clipmodel.encode_image(x, [24])
            patch_features = [
                self.clipmodel.visual._global_pool(t)[1] for t in patch_features
            ]
            patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
            patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
            return patch_features, cls_features
        else:
            raise ValueError("modality must be visual")

    def forward(self, x):
        image_input = x
        if not self.enable_image_adapter:
            cls_token, patch_tokens = self.clipmodel.encode_image(x, self.levels)
            cls_token = F.normalize(cls_token, dim=-1)
            patch_tokens = [self.clipmodel.visual.ln_post(t[:, 1:, :]) for t in patch_tokens]
            patch_tokens = [t @ self.clipmodel.visual.proj for t in patch_tokens]
            patch_tokens = [F.normalize(t, dim=-1) + cls_token.unsqueeze(1) for t in patch_tokens]
            return patch_tokens, cls_token

        # ========== patch embedding ==========
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # prepend CLS
        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        # (B,N,C) -> (L=N,B,C) for transformer
        x = x.permute(1, 0, 2)

        tokens = []
        for i in range(24):
            x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)

            if i < self.image_adapt_until:
                gate = torch.sigmoid(self.image_layer_gates[i])

                adapt_out = self.image_adapter["layer_adapters"][i](x)
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / (adapt_out.norm(dim=-1, keepdim=True) + 1e-6)
                )
                x = gate * adapt_out + (1.0 - gate) * x

            if i + 1 in self.levels:
                tokens.append(x[1:, :, :])  # remove CLS first

        # (L,B,C) -> (B,L,C)
        x = x.permute(1, 0, 2)
        tokens = [t.permute(1, 0, 2) for t in tokens]  # each scale: (B,N,C)

        # layer norm in image branch
        tokens = [self.image_encoder.ln_post(t) for t in tokens]

        # optional frozen DINOv3 dense-token residual fusion
        tokens = self._fuse_dinov3_into_tokens(tokens, image_input)

        # image-only 2D spatial high-pass on patch grids
        tokens = [self.image_adapter["spatial_hp"][i](t) for i, t in enumerate(tokens)]

        # seg branch: each scale projected + normalized
        seg_tokens = [
            self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
        ]
        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]

        # det branch: use the last scale after 2D spatial high-pass
        det_token = self.image_adapter["det_proj"](tokens[-1])
        det_token = F.normalize(det_token, dim=-1).mean(1)  # (B,C)

        return seg_tokens, det_token

    def forward_with_heatmap(self, x):
        seg_tokens, det_token = self.forward(x)

        scale_maps = []
        for t in seg_tokens:
            B, N, C = t.shape
            H = W = int(N ** 0.5)
            feat = t.mean(dim=-1).reshape(B, H, W)
            scale_maps.append(feat)

        multi = torch.stack(scale_maps, dim=1)
        fused = self.ms_fusion(multi)

        return seg_tokens, det_token, fused

    def encode_text(self, text, adapt_text=True):
        if (not self.enable_text_adapter) or (not adapt_text):
            return self.clipmodel.encode_text(text)

        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(cast_dtype)

        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)

        for i in range(12):
            x, attn = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )
            if i < self.text_adapt_until:
                gate = torch.sigmoid(self.text_layer_gates[i])

                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / (adapt_out.norm(dim=-1, keepdim=True) + 1e-6)
                )
                x = gate * adapt_out + (1.0 - gate) * x

        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x)

        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        return x
