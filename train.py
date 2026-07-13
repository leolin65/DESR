import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
try:
    import ipdb
except ImportError:
    ipdb = None
from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset
import dataset.constants as dataset_constants
from forward_utils import (
    get_adapted_text_embedding,
    get_adapted_single_class_text_embedding,
    calculate_similarity_map,
    calculate_seg_loss,
)
import warnings
import copy

# ✅ added/kept for memory control + AMP
import gc
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")



def _apply_data_path_override(args, logger=None):
    """Optionally override dataset root from CLI before get_dataset() is called.

    This avoids editing dataset/constants.py when running the same code on
    different machines. Example:
      --dataset MVTec --data_path "D:\\data\\mvtec_anomaly_detection"
    """
    data_path = getattr(args, "data_path", None)
    dataset_name = getattr(args, "dataset", None)
    if not dataset_name:
        return

    if data_path:
        root = os.path.abspath(os.path.expanduser(data_path))
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"--data_path does not exist or is not a directory: {root}"
            )
        dataset_constants.DATA_PATH[dataset_name] = root
        msg = f"Using overridden DATA_PATH[{dataset_name}] = {root}"
    else:
        root = dataset_constants.DATA_PATH.get(dataset_name, None)
        msg = f"Using DATA_PATH[{dataset_name}] = {root}"

    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

    if root is None:
        available = sorted(dataset_constants.DATA_PATH.keys())
        raise KeyError(
            f"Dataset {dataset_name} not found in DATA_PATH. Available: {available}"
        )
    if not os.path.isdir(root):
        raise FileNotFoundError(
            "Dataset root not found. Please fix dataset/constants.py or pass "
            f"--data_path. Current DATA_PATH[{dataset_name}] = {root}"
        )

    # Helpful MVTec-specific sanity check. It catches wrong parent folders early,
    # before DataLoader workers throw a long FileNotFoundError.
    if dataset_name == "MVTec":
        expected = os.path.join(root, "bottle")
        if not os.path.isdir(expected):
            raise FileNotFoundError(
                "MVTec root looks wrong. It should directly contain class folders "
                "such as bottle/, cable/, capsule/. Current root: "
                f"{root}"
            )

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_text_adapter(
    adapted_model: nn.Module,
    clip_surgery: nn.Module,
    text_norm_weight: float,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    start_epoch: int,
    save_path: str,
    text_epoch: int,
    dataset_name: str,
    img_size: int,
    logger: logging.Logger,
    text_anchor_sep_weight: float = 0.0,
    text_anchor_sep_max_cos: float = 0.20,
    text_anchor_preserve_weight: float = 0.0,
):
    
    adapted_model.train()
    adapted_model.clipmodel.eval()

    for epoch in range(start_epoch, text_epoch):
        logger.info(f"training text epoch {epoch}:")

        loss_list = []
        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            class_names = input_data["class_name"]

            # forward text
            epoch_text_feature_dict = {}
            epoch_base_text_feature_dict = {}
            use_base_anchor_preserve = float(text_anchor_preserve_weight) > 0.0

            for class_name in list(set(class_names)):
                text_embedding = get_adapted_single_class_text_embedding(
                    adapted_model, dataset_name, class_name, device
                )
                epoch_text_feature_dict[class_name] = text_embedding

                # Frozen CLIP text anchor. Used only for optional anchor-preservation
                # regularization, preventing the adapted prompts from drifting too far
                # away from CLIP's original normal/abnormal semantic space.
                if use_base_anchor_preserve:
                    with torch.no_grad():
                        base_text_embedding = get_adapted_single_class_text_embedding(
                            adapted_model.clipmodel, dataset_name, class_name, device
                        )
                    epoch_base_text_feature_dict[class_name] = base_text_embedding.detach()

            epoch_text_feature = torch.stack(
                [epoch_text_feature_dict[class_name] for class_name in class_names],
                dim=0,
            )  # bs,768,2

            epoch_base_text_feature = None
            if use_base_anchor_preserve:
                epoch_base_text_feature = torch.stack(
                    [epoch_base_text_feature_dict[class_name] for class_name in class_names],
                    dim=0,
                )  # bs,768,2

            # forward image
            with torch.no_grad():
                _, patch_features = clip_surgery.encode_image(image, [6, 12, 18, 24])
                cls_token, _ = adapted_model.clipmodel.encode_image(image, [])
                cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
                patch_features = [
                    clip_surgery.visual.ln_post(t[:, 1:, :]) for t in patch_features
                ]
                patch_features = [t @ clip_surgery.visual.proj for t in patch_features]
                patch_features = [
                    t / t.norm(dim=-1, keepdim=True) for t in patch_features
                ]
                patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]

            # # calculate similarity and get prediction
            # for f in patch_features:
            #     patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
            #     loss = calculate_seg_loss(patch_preds, mask)
            #     orthogonal_loss = (
            #         (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1])
            #         .sum(1)
            #         .mean()
            #     ) ** 2
            #     loss += orthogonal_loss * text_norm_weight

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # loss_list.append(loss.item())


            # calculate similarity and get prediction
            scale_losses = []
            for f in patch_features:
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                scale_losses.append(calculate_seg_loss(patch_preds, mask))

            seg_loss = torch.stack(scale_losses).mean()

            orthogonal_loss = (
                (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1])
                .sum(1)
                .mean()
            ) ** 2

            # ------------------------------------------------------------
            # V13 trainable add-on: text-anchor anti-collapse / preservation
            # ------------------------------------------------------------
            text_reg_loss = torch.zeros((), device=device)

            adapted_normal = F.normalize(epoch_text_feature[:, :, 0], dim=-1)
            adapted_abnormal = F.normalize(epoch_text_feature[:, :, 1], dim=-1)

            # Keep normal and abnormal anchors separated. This is more stable than
            # aggressive prompt banks because it only penalizes collapse.
            if float(text_anchor_sep_weight) > 0.0:
                cos_na = (adapted_normal * adapted_abnormal).sum(dim=-1)
                sep_loss = F.relu(cos_na - float(text_anchor_sep_max_cos)).mean()
                text_reg_loss = text_reg_loss + float(text_anchor_sep_weight) * sep_loss

            # Preserve the adapted anchors near frozen CLIP anchors to avoid hurting
            # zero-shot transfer and image-level calibration.
            if float(text_anchor_preserve_weight) > 0.0 and epoch_base_text_feature is not None:
                base_normal = F.normalize(epoch_base_text_feature[:, :, 0], dim=-1)
                base_abnormal = F.normalize(epoch_base_text_feature[:, :, 1], dim=-1)
                preserve_loss = (1.0 - (adapted_normal * base_normal).sum(dim=-1)).mean()
                preserve_loss = preserve_loss + (1.0 - (adapted_abnormal * base_abnormal).sum(dim=-1)).mean()
                text_reg_loss = text_reg_loss + float(text_anchor_preserve_weight) * preserve_loss

            loss = seg_loss + orthogonal_loss * text_norm_weight + text_reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())


        logger.info(f"loss: {np.mean(loss_list)}")
        ckp_path = os.path.join(save_path, "text_adapter.pth")
        # torch.save(
        #     {
        #         "epoch": epoch + 1,
        #         "text_adapter": adapted_model.text_adapter.state_dict(),
        #         "text_optimizer": optimizer.state_dict(),
        #     },
        #     ckp_path,
        # )

        torch.save(
            {
                "epoch": epoch + 1,
                "text_adapter": adapted_model.text_adapter.state_dict(),
                "text_layer_gates": adapted_model.text_layer_gates.detach().cpu(),
                "text_optimizer": optimizer.state_dict(),
            },
            ckp_path,
        )
                
    return adapted_model



def _clone_cpu_state_dict(module: nn.Module) -> dict:
    """Clone a module state_dict to CPU for EMA checkpointing."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _ema_update_state_dict(ema_state: dict, module: nn.Module, decay: float) -> dict:
    """Update a CPU EMA state from the current module state."""
    with torch.no_grad():
        cur_state = module.state_dict()
        for k, v in cur_state.items():
            v_cpu = v.detach().cpu()
            if k not in ema_state:
                ema_state[k] = v_cpu.clone()
            elif torch.is_floating_point(v_cpu):
                ema_state[k].mul_(decay).add_(v_cpu, alpha=(1.0 - decay))
            else:
                # integer/buffer-like entries: keep the current value
                ema_state[k] = v_cpu.clone()
    return ema_state


def _ema_update_tensor(ema_tensor: torch.Tensor, current_tensor: torch.Tensor, decay: float) -> torch.Tensor:
    with torch.no_grad():
        cur = current_tensor.detach().cpu()
        if ema_tensor is None:
            return cur.clone()
        return ema_tensor.mul(decay).add(cur, alpha=(1.0 - decay))


_CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=torch.float32).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=torch.float32).view(1, 3, 1, 1)


def _apply_hard_synth_anomaly(
    image: torch.Tensor,
    mask: torch.Tensor,
    label: torch.Tensor,
    synth_prob: float = 0.3,
    small_defect_prob: float = 0.3,
    low_contrast_prob: float = 0.3,
    scratch_prob: float = 0.0,
    blur_prob: float = 0.0,
):
    """V28 source-only hard synthetic anomaly.

    Operates on CLIP-normalized tensors and returns updated image/mask/label.
    The augmentation is deliberately simple and local: small rectangles, scratches,
    low-contrast stains, and optional blur-like smears.  It never uses target data.
    """
    prob = float(synth_prob)
    if prob <= 0.0:
        return image, mask, label

    B, C, H, W = image.shape
    device = image.device
    dtype = image.dtype
    mean = _CLIP_MEAN.to(device=device, dtype=dtype)
    std = _CLIP_STD.to(device=device, dtype=dtype)
    x01 = (image * std + mean).clamp(0.0, 1.0)
    new_mask = mask.clone()
    new_label = label.clone()

    for b in range(B):
        # Prefer adding synthetic defects to normal images, but also allow a small
        # chance on anomalous images to diversify masks.
        if torch.rand((), device=device).item() > prob:
            continue
        if int(label[b].detach().cpu().item()) != 0 and torch.rand((), device=device).item() > 0.25:
            continue

        is_small = torch.rand((), device=device).item() < float(small_defect_prob)
        if is_small:
            rh = int(torch.randint(max(3, H // 80), max(4, H // 28), (1,), device=device).item())
            rw = int(torch.randint(max(3, W // 80), max(4, W // 28), (1,), device=device).item())
        else:
            rh = int(torch.randint(max(4, H // 40), max(5, H // 12), (1,), device=device).item())
            rw = int(torch.randint(max(4, W // 40), max(5, W // 12), (1,), device=device).item())
        y0 = int(torch.randint(0, max(1, H - rh), (1,), device=device).item())
        x0 = int(torch.randint(0, max(1, W - rw), (1,), device=device).item())

        patch = x01[b:b+1, :, y0:y0+rh, x0:x0+rw]
        if patch.numel() == 0:
            continue

        low_contrast = torch.rand((), device=device).item() < float(low_contrast_prob)
        strength = 0.08 + 0.18 * torch.rand((), device=device).item() if low_contrast else 0.20 + 0.45 * torch.rand((), device=device).item()
        sign = -1.0 if torch.rand((), device=device).item() < 0.5 else 1.0
        noise = torch.randn_like(patch) * (0.03 if low_contrast else 0.08)
        color_shift = torch.empty((1, C, 1, 1), device=device, dtype=dtype).uniform_(-strength, strength)
        patch_aug = (patch + sign * color_shift + noise).clamp(0.0, 1.0)

        if torch.rand((), device=device).item() < float(scratch_prob):
            # Thin line/scratch inside the local box.
            yy = int(torch.randint(0, max(1, rh), (1,), device=device).item())
            thickness = max(1, rh // 10)
            patch_aug[:, :, yy:min(rh, yy+thickness), :] = (patch_aug[:, :, yy:min(rh, yy+thickness), :] * 0.35).clamp(0, 1)
        if torch.rand((), device=device).item() < float(blur_prob):
            patch_aug = F.avg_pool2d(patch_aug, kernel_size=3, stride=1, padding=1)

        x01[b:b+1, :, y0:y0+rh, x0:x0+rw] = patch_aug
        new_mask[b:b+1, :, y0:y0+rh, x0:x0+rw] = 1.0
        new_label[b] = 1

    image_aug = (x01 - mean) / std
    return image_aug, new_mask, new_label


def _make_ckpt_dict(model, optimizer, epoch, is_ema=False, ema_decay=None,
                    ema_image_adapter=None, ema_ms_fusion=None, ema_image_layer_gates=None,
                    ema_score_calibrator=None):
    if is_ema:
        image_adapter_state = ema_image_adapter
        ms_fusion_state = ema_ms_fusion
        image_layer_gates_state = ema_image_layer_gates
        score_calibrator_state = ema_score_calibrator
    else:
        image_adapter_state = model.image_adapter.state_dict()
        ms_fusion_state = model.ms_fusion.state_dict()
        image_layer_gates_state = model.image_layer_gates.detach().cpu()
        score_calibrator_state = model.score_calibrator.state_dict() if getattr(model, "score_calibrator", None) is not None else None
    out = {
        "epoch": epoch,
        "image_adapter": image_adapter_state,
        "ms_fusion": ms_fusion_state,
        "image_layer_gates": image_layer_gates_state,
        "image_optimizer": optimizer.state_dict() if optimizer is not None else None,
        "is_ema": bool(is_ema),
        "use_dinov3": bool(getattr(model, "use_dinov3", False)),
        "dino_model_name": getattr(model, "dino_model_name", None),
        "dino_fusion_alpha": float(getattr(model, "dino_fusion_alpha", 0.0)),
        "use_score_calibrator": bool(getattr(model, "use_score_calibrator", False)),
        "score_calib_alpha_min": float(getattr(model, "score_calib_alpha_min", 0.85)),
        "score_calib_alpha_max": float(getattr(model, "score_calib_alpha_max", 1.0)),
    }
    if score_calibrator_state is not None:
        out["score_calibrator"] = score_calibrator_state
    if ema_decay is not None:
        out["ema_decay"] = float(ema_decay)
    return out

def train_image_adapter(
    model: nn.Module,
    text_embeddings: torch.Tensor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    image_epoch: int,
    img_size: int,
    logger: logging.Logger,
    scaler: GradScaler,
    grad_accum: int = 1,
    use_adapter_ema: bool = False,
    adapter_ema_decay: float = 0.995,
    save_ema_as_eval: bool = True,
    use_score_calibrator: bool = False,
    image_calib_loss_weight: float = 0.0,
    score_rank_loss_weight: float = 0.0,
    score_rank_margin: float = 0.2,
    topk_ratio: float = 0.01,
    use_hard_synth_anomaly: bool = False,
    synth_prob: float = 0.0,
    synth_small_defect_prob: float = 0.3,
    synth_low_contrast_prob: float = 0.3,
    synth_scratch_prob: float = 0.0,
    synth_blur_prob: float = 0.0,
    save_best_source_checkpoint: bool = False,
):
    """
    ✅ Stage-2 memory-friendly training:
      - AMP autocast + GradScaler
      - Gradient accumulation
      - Explicit tensor cleanup to reduce fragmentation
    """
    model.train()
    model.clipmodel.eval()

    grad_accum = max(1, int(grad_accum))

    # V14 trainable add-on: adapter EMA.
    # If save_ema_as_eval=True, image_adapter_{epoch}.pth stores EMA weights so the
    # existing test.py can evaluate them directly. Raw checkpoints are also saved as
    # image_adapter_raw_{epoch}.pth for debugging.
    ema_decay = float(adapter_ema_decay)
    ema_image_adapter = _clone_cpu_state_dict(model.image_adapter) if use_adapter_ema else None
    ema_ms_fusion = _clone_cpu_state_dict(model.ms_fusion) if use_adapter_ema else None
    ema_image_layer_gates = model.image_layer_gates.detach().cpu().clone() if use_adapter_ema else None
    ema_score_calibrator = _clone_cpu_state_dict(model.score_calibrator) if (use_adapter_ema and getattr(model, "score_calibrator", None) is not None) else None
    best_source_proxy = None

    def update_ema_once():
        nonlocal ema_image_adapter, ema_ms_fusion, ema_image_layer_gates, ema_score_calibrator
        if not use_adapter_ema:
            return
        ema_image_adapter = _ema_update_state_dict(ema_image_adapter, model.image_adapter, ema_decay)
        ema_ms_fusion = _ema_update_state_dict(ema_ms_fusion, model.ms_fusion, ema_decay)
        ema_image_layer_gates = _ema_update_tensor(ema_image_layer_gates, model.image_layer_gates, ema_decay)
        if getattr(model, "score_calibrator", None) is not None:
            ema_score_calibrator = _ema_update_state_dict(ema_score_calibrator, model.score_calibrator, ema_decay)


    for epoch in range(start_epoch, image_epoch):
        logger.info(f"training image epoch {epoch}:")
        loss_list = []

        # ✅ important: zero_grad once per epoch (like your test.py style, reduce overhead)
        optimizer.zero_grad(set_to_none=True)
        step_idx = 0

        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device, non_blocking=True)
            mask = input_data["mask"].to(device, non_blocking=True)
            label = input_data["label"].to(device, non_blocking=True)

            B, C_img, H_img, W_img = image.shape

            if bool(use_hard_synth_anomaly):
                image, mask, label = _apply_hard_synth_anomaly(
                    image=image,
                    mask=mask,
                    label=label,
                    synth_prob=synth_prob,
                    small_defect_prob=synth_small_defect_prob,
                    low_contrast_prob=synth_low_contrast_prob,
                    scratch_prob=synth_scratch_prob,
                    blur_prob=synth_blur_prob,
                )

            # forward text
            class_names = input_data["class_name"]
            epoch_text_feature = torch.stack(
                [text_embeddings[class_name] for class_name in class_names], dim=0
            )

            # ------------------------------------------------------------
            # ✅ AMP forward + loss (this is the biggest memory saver)
            # ------------------------------------------------------------
            with autocast(enabled=scaler.is_enabled()):
                # forward image
                patch_features, det_feature = model(image)

                # ---------- Stage-2 image adapter with learnable multi-scale fusion ----------
                loss = 0.0

                # 1) image-level classification
                det_feature = det_feature.unsqueeze(1)                          # (B, 1, 768)
                cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0] # (B, 2)
                loss = loss + F.cross_entropy(cls_preds, label)
                cls_prob = torch.softmax(cls_preds, dim=1)[:, 1]

                # 2) patch-level multi-scale seg branch
                all_patch_preds = []
                for f in patch_features:
                    patch_pred = calculate_similarity_map(
                        f,
                        epoch_text_feature,
                        img_size,
                        test=False,
                    )  # (B, 2, H, W)
                    all_patch_preds.append(patch_pred)

                # stack → (B, S, 2, H, W)
                patch_stack = torch.stack(all_patch_preds, dim=1)  # (B, S, 2, H, W)
                B2, S, C2, H_s, W_s = patch_stack.shape

                # (B, S, 2, H, W) → (B, 2, S, H, W) → (B*2, S, H, W)
                patch_stack = patch_stack.permute(0, 2, 1, 3, 4).reshape(B2 * C2, S, H_s, W_s)

                fused = model.ms_fusion(patch_stack)      # (B*2, 1, H, W)
                fused = fused.view(B2, C2, H_s, W_s)      # (B, 2, H, W)

                loss = loss + calculate_seg_loss(fused, mask)

                # 3) V28 source-supervised dense-to-image score calibration.
                #    Uses only source labels/synthetic source anomalies during training.
                if bool(use_score_calibrator) and getattr(model, "score_calibrator", None) is not None and float(image_calib_loss_weight) > 0.0:
                    dense_map = fused[:, 1, :, :] - fused[:, 0, :, :]
                    final_score, alpha_i, dense_score, _ = model.calibrated_image_score(
                        dense_map,
                        cls_prob,
                        topk_ratio=topk_ratio,
                    )
                    final_score = final_score.clamp(1e-6, 1.0 - 1e-6)
                    label_float = label.float()
                    calib_loss = F.binary_cross_entropy(final_score, label_float)
                    loss = loss + float(image_calib_loss_weight) * calib_loss

                    if float(score_rank_loss_weight) > 0.0:
                        pos = final_score[label > 0]
                        neg = final_score[label == 0]
                        if pos.numel() > 0 and neg.numel() > 0:
                            rank_loss = F.relu(float(score_rank_margin) - pos.mean() + neg.mean())
                            loss = loss + float(score_rank_loss_weight) * rank_loss

            # track original loss (before grad_accum normalization)
            loss_value = float(loss.detach().float().cpu().item())

            # ------------------------------------------------------------
            # ✅ grad accumulation
            # ------------------------------------------------------------
            loss = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_idx += 1

            # step every grad_accum steps
            if step_idx % grad_accum == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_ema_once()

            loss_list.append(loss_value)

            # ------------------------------------------------------------
            # ✅ free big tensors ASAP to reduce fragmentation
            # ------------------------------------------------------------
            del image, mask, label, epoch_text_feature
            del patch_features, det_feature, cls_preds
            del all_patch_preds, patch_stack, fused
            gc.collect()

            # optional: occasionally empty cache (avoid too frequent)
            if (step_idx % 20) == 0:
                torch.cuda.empty_cache()

        # if the last steps didn't hit grad_accum boundary, still do a step
        if step_idx % grad_accum != 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update_ema_once()

        logger.info(f"loss: {np.mean(loss_list)}")

        # save checkpoint.  image_adapter_{epoch}.pth remains test.py-compatible.
        raw_model_dict = _make_ckpt_dict(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            is_ema=False,
        )

        if use_adapter_ema and save_ema_as_eval:
            model_dict = _make_ckpt_dict(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                is_ema=True,
                ema_decay=ema_decay,
                ema_image_adapter=ema_image_adapter,
                ema_ms_fusion=ema_ms_fusion,
                ema_image_layer_gates=ema_image_layer_gates,
                ema_score_calibrator=ema_score_calibrator,
            )
        else:
            model_dict = raw_model_dict

        torch.save(model_dict, os.path.join(save_path, "image_adapter.pth"))
        if (epoch + 1) % 1 == 0:
            ckp_path = os.path.join(save_path, f"image_adapter_{epoch + 1}.pth")
            torch.save(model_dict, ckp_path)
            if use_adapter_ema:
                raw_ckp_path = os.path.join(save_path, f"image_adapter_raw_{epoch + 1}.pth")
                torch.save(raw_model_dict, raw_ckp_path)

        # V28 source-specific checkpoint selection proxy.  This never uses target labels.
        # In this lightweight implementation, lower source training loss is used as the
        # source proxy, which is safer than target-specific epoch selection.
        if bool(save_best_source_checkpoint):
            source_proxy = float(np.mean(loss_list)) if len(loss_list) else float("inf")
            if best_source_proxy is None or source_proxy < best_source_proxy:
                best_source_proxy = source_proxy
                best_path = os.path.join(save_path, "image_adapter_best_source.pth")
                model_dict["source_selection_metric"] = "source_train_loss_proxy"
                model_dict["source_selection_value"] = float(source_proxy)
                torch.save(model_dict, best_path)
                logger.info("saved best source checkpoint: %s source_train_loss_proxy=%.6f", best_path, source_proxy)

    return model


def main():
    parser = argparse.ArgumentParser(description="Training")

    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="clip model to use (default: ViT-L-14-336)",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--surgery_until_layer", type=int, default=20)
    parser.add_argument("--relu", action="store_true", help="use relu after projection")

    # training
    parser.add_argument("--dataset", type=str, default="VisA")
    parser.add_argument("--data_path", type=str, default=None, help="Optional dataset root override, e.g. D:/data/mvtec_anomaly_detection")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers; use 0 for easier path debugging")
    parser.add_argument(
        "--training_mode",
        type=str,
        default="few_shot",
        choices=["few_shot", "full_shot"],
    )
    parser.add_argument("--shot", type=int, default=32, help="number of shots (0 means full shot)")
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--image_batch_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="DOS-friendly alias for --image_batch_size; if set, overrides --image_batch_size")
    parser.add_argument("--text_epoch", type=int, default=5, help="epochs for stage1")
    parser.add_argument("--image_epoch", type=int, default=20, help="epochs for stage2")
    parser.add_argument("--text_lr", type=float, default=0.00001, help="learning rate for stage1")
    parser.add_argument("--image_lr", type=float, default=0.0005, help="learning rate for stage2")
    parser.add_argument("--criterion", type=str, default=["dice_loss", "focal_loss"], nargs="+")

    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")

    # hyper-parameters
    parser.add_argument("--text_norm_weight", type=float, default=0.1)

    # V13 trainable add-on: text-anchor anti-collapse / preservation.
    # Defaults are 0.0, so the original V5/V11 training is exactly preserved unless enabled.
    parser.add_argument("--text_anchor_sep_weight", type=float, default=0.0,
                        help="penalty weight for preventing normal/abnormal text-anchor collapse")
    parser.add_argument("--text_anchor_sep_max_cos", type=float, default=0.20,
                        help="target maximum cosine similarity between normal and abnormal anchors")
    parser.add_argument("--text_anchor_preserve_weight", type=float, default=0.0,
                        help="penalty weight for preserving adapted anchors near frozen CLIP anchors")
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)

    # Optional frozen DINOv3 dense-token branch. Defaults keep original ICMR behavior.
    parser.add_argument("--use_dinov3", action="store_true", help="enable frozen DINOv3 dense-token fusion in the image branch")
    parser.add_argument("--dino_model_name", type=str, default="facebook/dinov3-vith16plus-pretrain-lvd1689m",
                        help="Hugging Face DINOv3 backbone name or local path")
    parser.add_argument("--dino_fusion_alpha", type=float, default=0.05,
                        help="residual DINOv3 -> CLIP token fusion strength; 0.0 disables fusion")
    parser.add_argument("--dino_img_size", type=int, default=None,
                        help="optional DINOv3 input resize; default uses the current image size")
    parser.add_argument("--dino_norm", type=str, default="lvd", choices=["lvd", "sat", "none"],
                        help="DINOv3 normalization: lvd for web-pretrained, sat for satellite-pretrained, none for raw [0,1]")
    parser.add_argument("--dino_local_files_only", action="store_true",
                        help="load DINOv3 only from local Hugging Face cache/path")
    parser.add_argument("--dino_residual_gate_mode", type=str, default="none",
                        choices=["none", "agreement", "token_norm"],
                        help="gate DINO residual by label-free dense reliability; none keeps original fixed residual")
    parser.add_argument("--dino_residual_gate_center", type=float, default=0.0,
                        help="center for DINO residual reliability gate")
    parser.add_argument("--dino_residual_gate_tau", type=float, default=0.25,
                        help="temperature for DINO residual reliability gate")
    parser.add_argument("--dino_residual_gate_min", type=float, default=0.0,
                        help="minimum DINO residual gate value in [0,1]; 0.5 is a conservative ablation choice")

    # ✅ NEW: minimal args for memory saving
    parser.add_argument("--amp", action="store_true", help="use mixed precision training")
    parser.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps")

    # V14 trainable add-on: save EMA-smoothed adapter checkpoints for evaluation.
    # This keeps inference cost exactly the same, unlike ensemble.
    parser.add_argument("--use_adapter_ema", action="store_true",
                        help="maintain EMA weights for image_adapter/ms_fusion/image_layer_gates")
    parser.add_argument("--adapter_ema_decay", type=float, default=0.995,
                        help="EMA decay for adapter weights; 0.99-0.997 are reasonable")
    parser.add_argument("--save_ema_as_eval", action="store_true",
                        help="save EMA weights as image_adapter_{epoch}.pth so test.py evaluates EMA directly")

    # V28 source-calibrated dense-to-image scoring.
    parser.add_argument("--use_score_calibrator", action="store_true",
                        help="train a small dense-to-image score calibrator for image-level scoring")
    parser.add_argument("--score_calib_alpha_min", type=float, default=0.85,
                        help="minimum dense-score weight predicted by the score calibrator")
    parser.add_argument("--score_calib_alpha_max", type=float, default=1.0,
                        help="maximum dense-score weight predicted by the score calibrator")
    parser.add_argument("--image_calib_loss_weight", type=float, default=0.0,
                        help="BCE loss weight for calibrated image-level score")
    parser.add_argument("--score_rank_loss_weight", type=float, default=0.0,
                        help="margin ranking loss weight for image-level scores")
    parser.add_argument("--score_rank_margin", type=float, default=0.2,
                        help="margin used by the optional image score ranking loss")
    parser.add_argument("--topk_ratio", type=float, default=0.01,
                        help="top-k ratio used by score calibrator during training")
    parser.add_argument("--use_hard_synth_anomaly", action="store_true",
                        help="enable source-only hard synthetic anomalies for score calibration")
    parser.add_argument("--synth_prob", type=float, default=0.0)
    parser.add_argument("--synth_small_defect_prob", type=float, default=0.3)
    parser.add_argument("--synth_low_contrast_prob", type=float, default=0.3)
    parser.add_argument("--synth_scratch_prob", type=float, default=0.0)
    parser.add_argument("--synth_blur_prob", type=float, default=0.0)
    parser.add_argument("--source_val_ratio", type=float, default=0.0,
                        help="accepted for protocol tracking; current best-source checkpoint uses source loss proxy")
    parser.add_argument("--select_checkpoint_by", type=str, default="source_loss",
                        choices=["source_loss", "source_s4"],
                        help="source-only checkpoint selection protocol tag; no target labels are used")
    parser.add_argument("--save_best_source_checkpoint", action="store_true",
                        help="save image_adapter_best_source.pth using a source-only proxy")

    parser.add_argument("--disable_text_adapter", action="store_true", help="run ablation without text adapter")
    parser.add_argument("--disable_image_adapter", action="store_true", help="run ablation without image adapter")

    args = parser.parse_args()
    if args.batch_size is not None:
        args.image_batch_size = int(args.batch_size)

    setup_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "train.log"),
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("args: %s", vars(args))
    _apply_data_path_override(args, logger)

    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # ✅ TF32 (optional)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ✅ AMP scaler
    scaler = GradScaler(enabled=(use_cuda and args.amp))

    # ========================================================
    # load model
    clip_surgery = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_surgery.eval()
    clip_surgery.visual.DAPM_replace(DPAM_layer=args.surgery_until_layer)

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
        enable_text_adapter=not args.disable_text_adapter,
        enable_image_adapter=not args.disable_image_adapter,
        use_dinov3=args.use_dinov3,
        dino_model_name=args.dino_model_name,
        dino_fusion_alpha=args.dino_fusion_alpha,
        dino_img_size=args.dino_img_size,
        dino_norm=args.dino_norm,
        dino_local_files_only=args.dino_local_files_only,
        dino_residual_gate_mode=args.dino_residual_gate_mode,
        dino_residual_gate_center=args.dino_residual_gate_center,
        dino_residual_gate_tau=args.dino_residual_gate_tau,
        dino_residual_gate_min=args.dino_residual_gate_min,
        use_score_calibrator=args.use_score_calibrator,
        score_calib_alpha_min=args.score_calib_alpha_min,
        score_calib_alpha_max=args.score_calib_alpha_max,
    ).to(device)
    model.eval()

    # set optimizer
    # text_optimizer = torch.optim.Adam(
    #     model.text_adapter.parameters(),
    #     lr=args.text_lr,
    #     betas=(0.5, 0.999),
    # )

    text_optimizer = None
    if not args.disable_text_adapter:
        text_optimizer = torch.optim.Adam(
            list(model.text_adapter.parameters()) + [model.text_layer_gates],
            lr=args.text_lr,
            betas=(0.5, 0.999),
        )

    # image_optimizer = torch.optim.Adam(
    #     list(model.image_adapter.parameters()) + list(model.ms_fusion.parameters()),
    #     lr=args.image_lr,
    #     betas=(0.5, 0.999),
    # )

    image_optimizer = None
    image_scheduler = None
    if not args.disable_image_adapter:
        image_optimizer = torch.optim.Adam(
            list(model.image_adapter.parameters())
            + list(model.ms_fusion.parameters())
            + ([p for p in model.score_calibrator.parameters()] if getattr(model, "score_calibrator", None) is not None else [])
            + [model.image_layer_gates],
            lr=args.image_lr,
            betas=(0.5, 0.999),
        )
        image_scheduler = MultiStepLR(image_optimizer, milestones=[16000, 32000], gamma=0.5)

    # ========================================================
    # load checkpoints if exists
    text_file = glob(args.save_path + "/text_adapter.pth")
    if args.disable_text_adapter:
        text_start_epoch = 0
        adapt_text = False
    elif len(text_file) > 0:
        checkpoint = torch.load(text_file[0])
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        if "text_layer_gates" in checkpoint:
            model.text_layer_gates.data.copy_(checkpoint["text_layer_gates"].to(device))
        if text_optimizer is not None and "text_optimizer" in checkpoint:
            text_optimizer.load_state_dict(checkpoint["text_optimizer"])
        text_start_epoch = checkpoint["epoch"]
        adapt_text = not (text_start_epoch == (args.text_epoch - 1))
    elif args.text_epoch == 0:
        text_start_epoch = 0
        adapt_text = False
    else:
        text_start_epoch = 0
        adapt_text = True

    file = glob(args.save_path + "/image_adapter.pth")
    if args.disable_image_adapter:
        image_start_epoch = 0
    elif len(file) > 0:
        checkpoint = torch.load(file[0])
        image_start_epoch = checkpoint["epoch"]
        image_state = checkpoint["image_adapter"]
        ckpt_has_dino = any(str(k).startswith("dinov3_proj") for k in image_state.keys())
        if ckpt_has_dino and (not args.use_dinov3):
            raise ValueError("This image_adapter checkpoint contains DINOv3 projection weights. Re-run train.py with --use_dinov3, or use a different --save_path.")
        if args.use_dinov3 and (not ckpt_has_dino):
            missing, unexpected = model.image_adapter.load_state_dict(image_state, strict=False)
            logger.warning("Loaded non-DINO image_adapter into DINOv3 model with strict=False. Missing keys: %s; unexpected keys: %s", missing, unexpected)
        else:
            model.image_adapter.load_state_dict(image_state)

        if "ms_fusion" in checkpoint:
            model.ms_fusion.load_state_dict(checkpoint["ms_fusion"])

        if "image_layer_gates" in checkpoint:
            model.image_layer_gates.data.copy_(checkpoint["image_layer_gates"].to(device))

        if getattr(model, "score_calibrator", None) is not None and "score_calibrator" in checkpoint:
            model.score_calibrator.load_state_dict(checkpoint["score_calibrator"])

        if image_optimizer is not None and "image_optimizer" in checkpoint:
            image_optimizer.load_state_dict(checkpoint["image_optimizer"])

    else:
        image_start_epoch = 0

    # ========================================================
    # load dataset
    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": int(args.num_workers), "pin_memory": True} if use_cuda else {"num_workers": int(args.num_workers)}
    logger.info("loading dataset ...")

    text_dataset, image_dataset = get_dataset(
        args.dataset,
        args.img_size,
        args.training_mode,
        args.shot,
        "train",
        logger,
    )

    text_dataloader = torch.utils.data.DataLoader(
        text_dataset, batch_size=args.text_batch_size, shuffle=True, **kwargs
    )

    logger.info("loading image adaptation dataset ...")
    image_dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
    )

    # ========================================================
    # training
    if adapt_text and (not args.disable_text_adapter):
        model = train_text_adapter(
            adapted_model=model,
            clip_surgery=clip_surgery,
            text_norm_weight=args.text_norm_weight,
            train_loader=text_dataloader,
            optimizer=text_optimizer,
            device=device,
            start_epoch=text_start_epoch,
            dataset_name=args.dataset,
            save_path=args.save_path,
            text_epoch=args.text_epoch,
            img_size=args.img_size,
            logger=logger,
            text_anchor_sep_weight=args.text_anchor_sep_weight,
            text_anchor_sep_max_cos=args.text_anchor_sep_max_cos,
            text_anchor_preserve_weight=args.text_anchor_preserve_weight,
        )

    del text_dataloader, text_dataset, clip_surgery, text_optimizer
    torch.cuda.empty_cache()

    with torch.no_grad():
        if args.disable_text_adapter or args.text_epoch == 0:
            text_embeddings = get_adapted_text_embedding(clip_model, args.dataset, device)
        else:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)

    if not args.disable_image_adapter:
        model = train_image_adapter(
            model=model,
            text_embeddings=text_embeddings,
            image_epoch=args.image_epoch,
            train_loader=image_dataloader,
            optimizer=image_optimizer,
            scheduler=image_scheduler,
            device=device,
            start_epoch=image_start_epoch,
            save_path=args.save_path,
            img_size=args.img_size,
            logger=logger,
            scaler=scaler,
            grad_accum=args.grad_accum,
            use_adapter_ema=args.use_adapter_ema,
            adapter_ema_decay=args.adapter_ema_decay,
            save_ema_as_eval=args.save_ema_as_eval,
            use_score_calibrator=args.use_score_calibrator,
            image_calib_loss_weight=args.image_calib_loss_weight,
            score_rank_loss_weight=args.score_rank_loss_weight,
            score_rank_margin=args.score_rank_margin,
            topk_ratio=args.topk_ratio,
            use_hard_synth_anomaly=args.use_hard_synth_anomaly,
            synth_prob=args.synth_prob,
            synth_small_defect_prob=args.synth_small_defect_prob,
            synth_low_contrast_prob=args.synth_low_contrast_prob,
            synth_scratch_prob=args.synth_scratch_prob,
            synth_blur_prob=args.synth_blur_prob,
            save_best_source_checkpoint=args.save_best_source_checkpoint,
        )
    else:
        logger.info("Skipping Stage-2 image adapter training (--disable_image_adapter).")


if __name__ == "__main__":
    main()