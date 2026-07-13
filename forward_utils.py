import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from kornia.filters import gaussian_blur2d
import ipdb
from dataset.constants import CLASS_NAMES, REAL_NAMES, PROMPTS
from model.tokenizer import tokenize
import pandas as pd
from dataset.constants import DATA_PATH
from utils import cos_sim

import gc
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc

# ================================================================================================
# The following code is used to get criterion for training


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(
        self,
        apply_nonlin=None,
        alpha=None,
        gamma=2,
        balance_index=0,
        smooth=1e-5,
        size_average=True,
    ):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (
            input_flat.sum(1) + targets_flat.sum(1) + smooth
        )
        loss = 1 - N_dice_eff.sum() / N
        return loss


# ================================================================================================
# The following code is used to get adapted text embeddings
prompt = PROMPTS
prompt_normal = prompt["prompt_normal"]
prompt_abnormal = prompt["prompt_abnormal"]
prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = prompt["prompt_templates"]


def get_adapted_single_class_text_embedding(model, dataset_name, class_name, device):
    if class_name == "object":
        real_name = class_name
    else:
        assert class_name in CLASS_NAMES[dataset_name], (
            f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
        )
        real_name = REAL_NAMES[dataset_name][class_name]
    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = class_embeddings / class_embeddings.norm(
            dim=-1, keepdim=True
        )
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        text_features.append(class_embedding)
    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features


def get_adapted_single_sentence_text_embedding(model, dataset_name, class_name, device):
    assert class_name in CLASS_NAMES[dataset_name], (
        f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
    )
    real_name = REAL_NAMES[dataset_name][class_name]
    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        text_features.append(class_embeddings)
    text_features = torch.cat(text_features, dim=0).to(device)
    return text_features


def get_adapted_text_embedding(model, dataset_name, device):
    ret_dict = {}
    for class_name in CLASS_NAMES[dataset_name]:
        text_features = get_adapted_single_class_text_embedding(
            model, dataset_name, class_name, device
        )
        ret_dict[class_name] = text_features
    return ret_dict



# ================================================================================================
def calculate_similarity_map(
    patch_features,
    epoch_text_feature,
    img_size,
    test=False,
    domain="Medical",
    fb_bg_suppression_beta: float = 0.0,
    fb_bg_suppression_mode: str = "normal_z",
    semantic_uncertainty_gamma: float = 0.0,
    semantic_uncertainty_tau: float = 0.5,
    semantic_uncertainty_logit_scale: float = 100.0,
    return_raw_map: bool = False,
):
    """
    patch_features: (B, N, C)
    epoch_text_feature:
        - 測試 / Stage-2: 通常是 (C, 2)
        - 訓練 / Stage-1: 可能是 (1, C, 2) 或其他 3 維 → 交給 matmul fallback

    V11 add-on:
        fb_bg_suppression_beta > 0 enables FB-CLIP-style background suppression
        on the pixel branch only. The original V5 map is still available through
        return_raw_map=True, so image-level top-k fusion can remain unchanged.

    V11b add-on:
        semantic_uncertainty_gamma > 0 suppresses uncertain patch responses where
        normal and abnormal text anchors are too close. This is applied only to the
        pixel map used for pixel metrics. The raw V5 map for image-level top-k
        fusion is intentionally left unchanged.
    """

    # ------------------------------------------------------------
    # 1) 儘量把 text feature 壓成 2 維 (C, 2)，方便使用 einsum
    # ------------------------------------------------------------
    if epoch_text_feature.dim() == 3:
        # 常見情況：shape = (1, C, 2) → squeeze 掉 batch 維
        if 1 in epoch_text_feature.shape:
            epoch_text_feature = epoch_text_feature.squeeze()
        # squeeze 完如果是 (C, 2)，就等同原本設計

    # ------------------------------------------------------------
    # 2) 計算 patch_anomaly_scores
    #    - 若為 2D → 用 einsum 加速（等價於 matmul）
    #    - 其他情況 → 回退到原本 matmul，避免 shape 不合造成錯誤
    # ------------------------------------------------------------
    if epoch_text_feature.dim() == 2:
        # patch_features: (B, N, C)
        # epoch_text_feature: (C, 2)
        # 結果: (B, N, 2)
        patch_anomaly_scores = 100.0 * torch.einsum(
            "bnc,cd->bnd", patch_features, epoch_text_feature
        )
    else:
        # 保險：維度不標準時，用原本 matmul 行為
        patch_anomaly_scores = 100.0 * torch.matmul(
            patch_features, epoch_text_feature
        )

    B, L, C = patch_anomaly_scores.shape
    H = int(np.sqrt(L))

    # (B, C, H, H)，C=2 時就是 normal / anomaly 兩個 channel
    patch_pred = patch_anomaly_scores.permute(0, 2, 1).view(B, C, H, H)

    # ------------------------------------------------------------
    # 3) AA-CLIP 原始的 (A - N) + Gaussian blur
    # ------------------------------------------------------------
    if test:
        assert C == 2
        sigma = 1 if domain == "Industrial" else 1.5
        kernel_size = 7 if domain == "Industrial" else 9

        # ------------------------------------------------------------
        # V5 raw pixel anomaly map: abnormal logit - normal/background logit
        # ------------------------------------------------------------
        normal_logit = patch_pred[:, 0]    # (B,H,W)
        abnormal_logit = patch_pred[:, 1]  # (B,H,W)
        raw_pixel_map = abnormal_logit - normal_logit

        # ------------------------------------------------------------
        # V11 FB-CLIP-style background suppression on pixel branch only
        #   - normal_z: suppress spatial locations with unusually high normal/background evidence
        #   - normal_logit: direct logit suppression, stronger and less stable
        #   - margin_gate: soft gate using normal probability, conservative
        # ------------------------------------------------------------
        beta = float(fb_bg_suppression_beta)
        if beta > 0:
            mode = str(fb_bg_suppression_mode).lower()
            if mode == "normal_z":
                bg = normal_logit
                bg_mean = bg.flatten(1).mean(dim=1).view(-1, 1, 1)
                bg_std = bg.flatten(1).std(dim=1).view(-1, 1, 1).clamp_min(1e-6)
                bg_conf = F.relu((bg - bg_mean) / bg_std)
                patch_pred = raw_pixel_map - beta * bg_conf
            elif mode == "normal_logit":
                patch_pred = raw_pixel_map - beta * normal_logit
            elif mode == "margin_gate":
                prob = torch.softmax(torch.stack([normal_logit, abnormal_logit], dim=1), dim=1)
                bg_prob = prob[:, 0]
                patch_pred = raw_pixel_map * (1.0 - beta * bg_prob)
            else:
                raise ValueError(f"Unknown fb_bg_suppression_mode: {fb_bg_suppression_mode}")
        else:
            patch_pred = raw_pixel_map

        # ------------------------------------------------------------
        # V11b Semantic-Uncertainty Suppression on pixel branch only
        #   If normal/abnormal text anchors are close for a patch, the
        #   patch is semantically uncertain and should contribute less
        #   to the anomaly heatmap.
        #
        #   NOTE: patch_anomaly_scores above are multiplied by 100.0.
        #   Therefore we normalize raw_pixel_map by
        #   semantic_uncertainty_logit_scale so that tau=0.5 has the
        #   intended CLIP-cosine-scale meaning.
        # ------------------------------------------------------------
        gamma_unc = float(semantic_uncertainty_gamma)
        if gamma_unc > 0:
            tau_unc = max(float(semantic_uncertainty_tau), 1e-6)
            logit_scale = max(float(semantic_uncertainty_logit_scale), 1e-6)
            semantic_margin = raw_pixel_map / logit_scale
            uncertainty = torch.exp(-torch.abs(semantic_margin) / tau_unc)
            uncertainty_gate = torch.clamp(1.0 - gamma_unc * uncertainty, min=0.0, max=1.0)
            patch_pred = patch_pred * uncertainty_gate

        patch_pred = gaussian_blur2d(
            patch_pred.unsqueeze(1),
            (kernel_size, kernel_size),
            (sigma, sigma),
        )

        if return_raw_map:
            raw_pixel_map = gaussian_blur2d(
                raw_pixel_map.unsqueeze(1),
                (kernel_size, kernel_size),
                (sigma, sigma),
            )
            raw_pixel_map = F.interpolate(
                raw_pixel_map, size=img_size, mode="bilinear", align_corners=True
            )

    patch_preds = F.interpolate(
        patch_pred, size=img_size, mode="bilinear", align_corners=True
    )

    if not test and C > 1:
        patch_preds = torch.softmax(patch_preds, dim=1)

    if test and return_raw_map:
        return patch_preds, raw_pixel_map

    return patch_preds



focal_loss = FocalLoss()
dice_loss = BinaryDiceLoss()


def calculate_seg_loss(patch_preds, mask):
    loss = focal_loss(patch_preds, mask)
    loss += dice_loss(patch_preds[:, 0, :, :], 1 - mask)
    loss += dice_loss(patch_preds[:, 1, :, :], mask)
    return loss



def _binary_auc_no_binarize(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Exact ROC-AUC for binary labels, but avoids roc_auc_score()'s internal
    label_binarize(...).toarray() path that can explode RAM on huge N.
    """
    # force GT to {0,1} (RS masks often are 0/255)
    y_true = (y_true > 0).astype(np.uint8, copy=False)
    # reduce peak memory
    y_score = y_score.astype(np.float32, copy=False)

    if np.unique(y_true).size < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    return float(auc(fpr, tpr))




def _compute_aupro(
    pixel_label: np.ndarray,
    pixel_preds: np.ndarray,
    max_fpr: float = 0.30,
    num_thresholds: int = 200,
) -> float:
    """
    Compute pixel-level AUPRO/PRO for anomaly segmentation.

    This follows the common industrial AD protocol: for each threshold, compute
    the per-region overlap (PRO) over connected GT anomaly components, compute
    pixel-level false-positive rate over normal pixels, and integrate the PRO-FPR
    curve up to max_fpr. Output is in percentage [0, 100].

    Args:
        pixel_label: (N,1,H,W) or (N,H,W), nonzero means anomaly.
        pixel_preds: (N,H,W) or (N,1,H,W), anomaly scores already normalized or raw.
        max_fpr: upper FPR bound for normalized partial AUC, usually 0.30.
        num_thresholds: number of thresholds used to sample the PRO curve.
    """
    if pixel_label is None or pixel_preds is None or pixel_preds.size == 0:
        return 0.0

    gt = np.asarray(pixel_label)
    pred = np.asarray(pixel_preds)

    if gt.ndim == 4:
        gt = gt[:, 0]
    if pred.ndim == 4:
        pred = pred[:, 0]

    gt = (gt > 0).astype(np.uint8, copy=False)
    pred = pred.astype(np.float32, copy=False)

    # Need both normal and anomalous pixels for FPR/PRO to be meaningful.
    if gt.max() == 0 or gt.min() == 1:
        return 0.0

    p_min = float(pred.min())
    p_max = float(pred.max())
    if p_max > p_min:
        pred = (pred - p_min) / (p_max - p_min + 1e-8)
    else:
        return 0.0

    # Pre-compute connected anomaly regions once.
    regions = []
    for i in range(gt.shape[0]):
        num_labels, labels = cv2.connectedComponents(gt[i].astype(np.uint8), connectivity=8)
        for rid in range(1, num_labels):
            region_mask = labels == rid
            area = int(region_mask.sum())
            if area > 0:
                regions.append((i, region_mask, area))

    if len(regions) == 0:
        return 0.0

    normal_mask = gt == 0
    total_normal = int(normal_mask.sum())
    if total_normal == 0:
        return 0.0

    thresholds = np.linspace(1.0, 0.0, int(num_thresholds), dtype=np.float32)
    pros = []
    fprs = []

    for thr in thresholds:
        pred_bin = pred >= float(thr)

        fp = int((pred_bin & normal_mask).sum())
        fpr = fp / float(total_normal)

        overlaps = []
        for img_idx, region_mask, area in regions:
            overlap = int((pred_bin[img_idx] & region_mask).sum()) / float(area)
            overlaps.append(overlap)

        pros.append(float(np.mean(overlaps)))
        fprs.append(float(fpr))

    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)

    # Sort and keep FPR <= max_fpr; add boundary points for stable integration.
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    keep = fprs <= float(max_fpr)
    fprs_kept = fprs[keep]
    pros_kept = pros[keep]

    if fprs_kept.size == 0:
        fprs_kept = np.array([0.0], dtype=np.float64)
        pros_kept = np.array([pros[0]], dtype=np.float64)

    # Ensure curve starts at FPR=0.
    if fprs_kept[0] > 0:
        fprs_kept = np.concatenate([[0.0], fprs_kept])
        pros_kept = np.concatenate([[pros_kept[0]], pros_kept])

    # Interpolate PRO at max_fpr if needed.
    if fprs_kept[-1] < float(max_fpr):
        pro_at_max = np.interp(float(max_fpr), fprs, pros)
        fprs_kept = np.concatenate([fprs_kept, [float(max_fpr)]])
        pros_kept = np.concatenate([pros_kept, [pro_at_max]])
    elif fprs_kept[-1] > float(max_fpr):
        pro_at_max = np.interp(float(max_fpr), fprs_kept, pros_kept)
        valid = fprs_kept < float(max_fpr)
        fprs_kept = np.concatenate([fprs_kept[valid], [float(max_fpr)]])
        pros_kept = np.concatenate([pros_kept[valid], [pro_at_max]])

    aupro = auc(fprs_kept, pros_kept) / float(max_fpr)
    return round(float(aupro) * 100.0, 4)

def metrics_eval(
    pixel_label: np.ndarray,
    image_label: np.ndarray,
    pixel_preds: np.ndarray,
    image_preds: np.ndarray,
    class_names: str,
    domain: str,
    pixel_preds_for_image = None,
    image_fusion_alpha: float = 0.7,
    topk_ratio: float = 0.02,
):
    """
    pixel_label: (N, 1, H, W) or (N, H, W) or possibly dummy if no pixel GT
    image_label: (N,)
    pixel_preds: (N, H, W) or (N, 1, H, W) -- used for pixel-level metrics
    image_preds: (N,)  -- cls-based image score (before reweight)
    pixel_preds_for_image: optional raw V5 pixel map used only for image top-k fusion.
        This keeps V11 background suppression strictly pixel-branch only.
    """

    # 轉成 numpy 保險一點
    pixel_label = np.asarray(pixel_label)
    image_label = np.asarray(image_label)
    pixel_preds = np.asarray(pixel_preds)
    image_preds = np.asarray(image_preds)
    pixel_preds_for_image = pixel_preds if pixel_preds_for_image is None else np.asarray(pixel_preds_for_image)

    # ----------------------------------------------------------------------
    # 0) 正規化到 [0,1]，避免 scale 差太多
    # ----------------------------------------------------------------------
    if pixel_preds.size > 0:
        p_min, p_max = pixel_preds.min(), pixel_preds.max()
        if p_max > p_min:
            pixel_preds = (pixel_preds - p_min) / (p_max - p_min + 1e-8)

    if pixel_preds_for_image.size > 0:
        pi_min, pi_max = pixel_preds_for_image.min(), pixel_preds_for_image.max()
        if pi_max > pi_min:
            pixel_preds_for_image = (pixel_preds_for_image - pi_min) / (pi_max - pi_min + 1e-8)

    if image_preds.size > 0:
        i_min, i_max = image_preds.min(), image_preds.max()
        if i_max > i_min:
            image_preds = (image_preds - i_min) / (i_max - i_min + 1e-8)

    # ----------------------------------------------------------------------
    # 1) top-k pooling：取每張圖 top K% pixel 的平均當 per-image anomaly score
    # ----------------------------------------------------------------------
    TOPK_RATIO = float(topk_ratio)
    IMAGE_FUSION_ALPHA = float(image_fusion_alpha)

    # Important for V11: image-level top-k uses pixel_preds_for_image, not necessarily
    # the background-suppressed pixel map used for pixel AUROC/AP/F1/IoU.
    if pixel_preds_for_image.ndim == 4:
        # (N,1,H,W) or (N,C,H,W) -> 合併 channel
        N = pixel_preds_for_image.shape[0]
        flat = pixel_preds_for_image.reshape(N, -1)
    elif pixel_preds_for_image.ndim == 3:
        # (N,H,W)
        N = pixel_preds_for_image.shape[0]
        flat = pixel_preds_for_image.reshape(N, -1)
    else:
        # 沒有 pixel map 的極端情況（理論上不會在你現在的 pipeline 發生）
        flat = None
        N = image_preds.shape[0]

    if flat is not None and flat.shape[1] > 0:
        num_pixels = flat.shape[1]
        k = max(1, int(num_pixels * TOPK_RATIO))

        # np.argpartition 比 sort 快，只抓出 top-k
        topk_idx = np.argpartition(flat, -k, axis=1)[:, -k:]   # (N, k)
        topk_vals = np.take_along_axis(flat, topk_idx, axis=1) # (N, k)
        topk_mean = topk_vals.mean(axis=1)                     # (N,)
    else:
        # fallback：不用 pixel reweight
        topk_mean = None

    # ----------------------------------------------------------------------
    # 2) image-level reweight
    # ----------------------------------------------------------------------
    if topk_mean is not None:
        if domain != "Medical":
            # 工業：alpha * topk_mean + (1-alpha) * cls_score
            image_score = IMAGE_FUSION_ALPHA * topk_mean + (1.0 - IMAGE_FUSION_ALPHA) * image_preds
        else:
            # 醫療：直接用 topk_mean（比單點 pmax 穩）
            image_score = topk_mean
    else:
        # 沒有 pixel map 的極端情況，就只用原本 image_preds
        image_score = image_preds

    # # ----------------------------------------------------------------------
    # # 3) pixel level auc & ap（保留，但避免沒有 GT 時報錯）
    # # ----------------------------------------------------------------------
    # # pixel_label 可能是 (N,1,H,W) or (N,H,W) 或 dummy 全 0
    # if pixel_preds.size > 0:
    #     pixel_label_flat = pixel_label.flatten()
    #     pixel_preds_flat = pixel_preds.flatten()

    #     # 如果 pixel GT 只有一種 label（例如全 0），roc_auc 會 throw exception
    #     if np.unique(pixel_label_flat).size > 1:
    #         zero_pixel_auc = roc_auc_score(pixel_label_flat, pixel_preds_flat)
    #         zero_pixel_ap = average_precision_score(pixel_label_flat, pixel_preds_flat)
    #         pixel_auc_out = round(zero_pixel_auc, 4) * 100
    #         pixel_ap_out = round(zero_pixel_ap, 4) * 100
    #     else:
    #         pixel_auc_out = 0.0
    #         pixel_ap_out = 0.0
    # else:
    #     pixel_auc_out = 0.0
    #     pixel_ap_out = 0.0

    # ----------------------------------------------------------------------
    # 3) pixel level auc & ap（精確 AUC、避免 roc_auc_score→label_binarize→toarray 爆 RAM）
    # ----------------------------------------------------------------------
    if pixel_preds.size > 0:
        pixel_label_flat = pixel_label.flatten()
        pixel_preds_flat = pixel_preds.flatten()

        # ✅ 強制 GT 變成 {0,1}，避免 0/255 造成 sklearn 走怪分支
        pixel_label_flat = (pixel_label_flat > 0).astype(np.uint8, copy=False)
        # ✅ preds 用 float32，減少中間陣列壓力
        pixel_preds_flat = pixel_preds_flat.astype(np.float32, copy=False)

        if np.unique(pixel_label_flat).size > 1:
            # ✅ 用 roc_curve+auc（精確 AUC），避開 roc_auc_score 的 toarray 路徑
            zero_pixel_auc = _binary_auc_no_binarize(pixel_label_flat, pixel_preds_flat)

            # ✅ AP 保留 average_precision_score（通常不會走 (N,2) dense toarray）
            zero_pixel_ap = average_precision_score(pixel_label_flat, pixel_preds_flat)

            pixel_auc_out = round(zero_pixel_auc, 4) * 100
            pixel_ap_out = round(zero_pixel_ap, 4) * 100
        else:
            pixel_auc_out = 0.0
            pixel_ap_out = 0.0

        # ✅ 提早釋放巨型 flatten array，避免 epoch2 RAM 累積
        del pixel_label_flat, pixel_preds_flat
        gc.collect()
    else:
        pixel_auc_out = 0.0
        pixel_ap_out = 0.0

    # ----------------------------------------------------------------------
    # 4) pixel-level PRO / AUPRO, normalized partial AUC up to FPR=0.30
    # ----------------------------------------------------------------------
    pro_out = 0.0
    if pixel_preds.size > 0:
        try:
            pro_out = _compute_aupro(pixel_label, pixel_preds, max_fpr=0.30, num_thresholds=200)
        except Exception as e:
            # Do not break full evaluation if a dataset has unusual masks.
            print(f"Warning: PRO/AUPRO computation failed for {class_names}: {e}")
            pro_out = 0.0
        gc.collect()
        
    # ----------------------------------------------------------------------
    # 5) image level auc & ap
    # ----------------------------------------------------------------------
    image_label_flat = image_label.flatten()
    if np.unique(image_label_flat).size > 1:
        agg_image_preds = image_score.flatten()
        agg_image_auc = roc_auc_score(image_label_flat, agg_image_preds)
        agg_image_ap = average_precision_score(image_label_flat, agg_image_preds)
        image_auc_out = round(agg_image_auc, 4) * 100
        image_ap_out = round(agg_image_ap, 4) * 100
    else:
        image_auc_out = 0.0
        image_ap_out = 0.0

    # result = {
    #     "class name": class_names,
    #     "pixel AUC": pixel_auc_out,
    #     "pixel AP": pixel_ap_out,
    #     "image AUC": image_auc_out,
    #     "image AP": image_ap_out,
    # }
    # return result


    # ----------------------------------------------------------------------
    # 6) threshold-based segmentation metrics (F1 / IoU)  [RS change detection friendly]
    # ----------------------------------------------------------------------
    # NOTE:
    # - 不做 stride/downsample
    # - 不改 img_size=518
    # - 這裡用固定閾值 0.5（常見 CD baseline）；若你之後要換成 Otsu/Best-F1，也可以在這段改

    f1_out = 0.0
    iou_out = 0.0

    if pixel_preds.size > 0:
        # pixel_label: (N,1,H,W) or (N,H,W) ; pixel_preds: (N,H,W)  (你的程式流通常是這樣)
        gt = pixel_label
        if gt.ndim == 4:
            gt = gt[:, 0]  # (N,H,W)
        gt = (gt > 0).astype(np.uint8, copy=False)

        pr = pixel_preds.astype(np.float32, copy=False)

        # normalize preds to [0,1] to make threshold meaningful across epochs
        pr_min = float(pr.min())
        pr_max = float(pr.max())
        if pr_max > pr_min:
            pr = (pr - pr_min) / (pr_max - pr_min)
        else:
            pr = np.zeros_like(pr, dtype=np.float32)

        thr = 0.5
        pred_bin = (pr >= thr).astype(np.uint8, copy=False)

        # compute TP/FP/FN over ALL pixels (no downsample)
        tp = int(((pred_bin == 1) & (gt == 1)).sum())
        fp = int(((pred_bin == 1) & (gt == 0)).sum())
        fn = int(((pred_bin == 0) & (gt == 1)).sum())

        denom_f1 = (2 * tp + fp + fn)
        denom_iou = (tp + fp + fn)

        if denom_f1 > 0:
            f1_out = (2.0 * tp) / float(denom_f1) * 100.0
        if denom_iou > 0:
            iou_out = (tp) / float(denom_iou) * 100.0

        # free big arrays early
        del gt, pr, pred_bin
        gc.collect()

    result = {
        "class name": class_names,
        "pixel AUC": pixel_auc_out,
        "pixel AP": pixel_ap_out,
        "PRO": round(pro_out, 4),
        "F1": round(f1_out, 4),
        "IoU": round(iou_out, 4),
        "image AUC": image_auc_out,
        "image AP": image_ap_out,
    }
    return result


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    return (alpha * image + (1 - alpha) * scoremap).astype(np.uint8)


# def visualize(
#     pixel_label: np.ndarray,
#     pixel_preds: np.ndarray,
#     file_names: list[str],
#     save_dir: str,
#     dataset_name: str,
#     class_name: str,
# ):
#     if pixel_preds.max() != 1:
#         pixel_preds = (pixel_preds - pixel_preds.min()) / (
#             pixel_preds.max() - pixel_preds.min()
#         )
#         pixel_preds = (pixel_preds * 255).astype(np.uint8)
#     if pixel_label.dtype != np.uint8:
#         pixel_label = pixel_label != 0
#         pixel_label = (pixel_label * 255).astype(np.uint8)
#     # ===============================================================================================
#     # save path
#     save_dir = os.path.join(save_dir, "visualization", dataset_name, class_name)
#     os.makedirs(save_dir, exist_ok=True)
#     for idx, file in enumerate(file_names):
#         image_file = os.path.join(DATA_PATH[dataset_name], file)
#         image = cv2.imread(image_file)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, pixel_label.shape[-2:])
#         save_image_list = [image]

#         if dataset_name == "MVTec":
#             damage_name, image_name = file.split("/")[-2:]
#             file_name = f"{damage_name}_{image_name}"
#         else:
#             raise NotImplementedError

#         save_image_list.append(cv2.cvtColor(pixel_label[idx, 0], cv2.COLOR_GRAY2RGB))
#         save_image_list.append(cv2.cvtColor(pixel_preds[idx], cv2.COLOR_GRAY2RGB))
#         save_image_list = save_image_list[:1] + [
#             apply_ad_scoremap(image, _) for _ in save_image_list[1:]
#         ]
#         scoremap = np.vstack(save_image_list)
#         cv2.imwrite(os.path.join(save_dir, file_name), scoremap)
def _draw_gt_red_contour(image_rgb, gt_mask, thickness=2):
    """
    Draw GT as red contour only.
    This does NOT fill the GT region.

    Args:
        image_rgb: RGB uint8 image, shape [H, W, 3].
        gt_mask: binary or 0/255 mask, shape [H, W].
        thickness: contour line width.

    Returns:
        RGB uint8 image with red GT outline.
    """
    out = image_rgb.copy()

    if gt_mask is None:
        return out

    gt = np.asarray(gt_mask)
    gt = np.squeeze(gt)

    if gt.ndim != 2:
        return out

    # Convert GT to binary mask.
    # Supports both {0,1} and {0,255} masks.
    if gt.max() > 1:
        gt_bin = (gt > 127).astype(np.uint8)
    else:
        gt_bin = (gt > 0).astype(np.uint8)

    if gt_bin.sum() == 0:
        return out

    h, w = out.shape[:2]
    if gt_bin.shape[:2] != (h, w):
        gt_bin = cv2.resize(gt_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(
        gt_bin,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # RGB red. Since the image is RGB here, use (255, 0, 0).
    cv2.drawContours(out, contours, -1, (255, 0, 0), thickness)
    return out


def visualize(
    pixel_label,
    pixel_preds,
    file_names,
    save_dir,
    dataset_name,
    class_name,
):
    """
    Save visualization as:
        Original image / GT red outline / Heatmap + GT red outline

    Important:
        - GT is drawn only as a red contour.
        - GT is NOT filled as a red mask.
        - Heatmap is overlaid with GT red outline for paper-quality visualization.
    """
    # -------------------------------------------------------------------------
    # 1. Normalize prediction maps to uint8 [0, 255]
    # -------------------------------------------------------------------------
    pixel_preds = np.asarray(pixel_preds)
    pred_min = float(pixel_preds.min())
    pred_max = float(pixel_preds.max())

    if pred_max - pred_min > 1e-8:
        pixel_preds_vis = (pixel_preds - pred_min) / (pred_max - pred_min + 1e-8)
    else:
        pixel_preds_vis = np.zeros_like(pixel_preds, dtype=np.float32)

    pixel_preds_vis = (pixel_preds_vis * 255).clip(0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # 2. Convert GT to uint8 binary mask [0, 255]
    # -------------------------------------------------------------------------
    pixel_label = np.asarray(pixel_label)
    if pixel_label.dtype != np.uint8:
        pixel_label_vis = (pixel_label != 0).astype(np.uint8) * 255
    else:
        # keep 0/255 style even if input is uint8 but in {0,1}
        if pixel_label.max() <= 1:
            pixel_label_vis = (pixel_label != 0).astype(np.uint8) * 255
        else:
            pixel_label_vis = pixel_label.copy()

    # -------------------------------------------------------------------------
    # 3. Save directory
    # -------------------------------------------------------------------------
    save_dir = os.path.join(save_dir, "visualization", dataset_name, class_name)
    os.makedirs(save_dir, exist_ok=True)

    for idx, file in enumerate(file_names):
        # ---------------------------------------------------------------------
        # 4. Read original image
        # ---------------------------------------------------------------------
        image_file = os.path.join(DATA_PATH[dataset_name], file)
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

        if image is None:
            # Fallback for special tif/tiff packaging
            try:
                from PIL import Image
                image = np.array(Image.open(image_file).convert("RGB"))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception as exc:
                print(f"Warning: Could not read image {image_file}: {exc}")
                continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Match image size to mask/prediction size.
        # pixel_label_vis usually has shape [N, 1, H, W].
        target_h = pixel_label_vis.shape[-2]
        target_w = pixel_label_vis.shape[-1]
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # ---------------------------------------------------------------------
        # 5. Prepare GT and heatmap
        # ---------------------------------------------------------------------
        if pixel_label_vis.ndim == 4:
            gt_mask = pixel_label_vis[idx, 0]
        elif pixel_label_vis.ndim == 3:
            gt_mask = pixel_label_vis[idx]
        else:
            gt_mask = pixel_label_vis

        pred_map = pixel_preds_vis[idx]
        if pred_map.shape[:2] != image.shape[:2]:
            pred_map = cv2.resize(pred_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Column 1: original image
        image_only = image

        # Column 2: original image + GT red outline only
        gt_outline = _draw_gt_red_contour(image, gt_mask, thickness=2)

        # Column 3: heatmap overlay + GT red outline
        heatmap = cv2.applyColorMap(pred_map, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_overlay = cv2.addWeighted(image, 0.50, heatmap, 0.50, 0)
        heatmap_with_gt_outline = _draw_gt_red_contour(heatmap_overlay, gt_mask, thickness=2)

        # ---------------------------------------------------------------------
        # 6. File naming
        # ---------------------------------------------------------------------
        if dataset_name == "MVTec":
            damage_name, image_name = file.split("/")[-2:]
            file_name = f"{damage_name}_{os.path.splitext(image_name)[0]}"
        else:
            base_name = os.path.basename(file)
            file_name = os.path.splitext(base_name)[0]

        # ---------------------------------------------------------------------
        # 7. Save vertical panel
        # ---------------------------------------------------------------------
        final_img = np.vstack([image_only, gt_outline, heatmap_with_gt_outline])
        final_save_path = os.path.join(save_dir, f"{file_name}.png")

        # cv2.imwrite expects BGR
        cv2.imwrite(final_save_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
