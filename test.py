import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import re

from utils import setup_seed, cos_sim
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS
import dataset.constants as dataset_constants
from forward_utils import (
    get_adapted_text_embedding,
    calculate_similarity_map,
    metrics_eval,
    visualize,
)
import warnings
import gc
import shutil
from sklearn.metrics import roc_auc_score, average_precision_score


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


def get_support_features(model, support_loader, device):  #raise NotImplementedError("get_support_features() is outdated and should not be used.")
    
    all_features = []
    for input_data in support_loader:  # bs always=1. training for an epoch first, Then use this updated model for memory bank construction.
        image = input_data[0].to(device)
        patch_tokens = model(image)
        patch_tokens = [t.reshape(-1, 768) for t in patch_tokens]
        all_features.append(patch_tokens)
    support_features = [
        torch.cat([all_features[j][i] for j in range(len(all_features))], dim=0)
        for i in range(len(all_features[0]))
    ]
    return support_features


# ================================================================================================
# V22: image-aware reliable top-k fusion
# --------------------------------------------------------------------------------
# This is a test-time, image-branch-only modification.  Pixel metrics still use the
# V11/V13+V14 pixel map.  Image-level top-k fusion uses:
#   reliable_raw_map = raw_pixel_map * (1 + rho * tanh(|abnormal-normal|/scale/tau))
# It borrows the semantic uncertainty idea from V11b, but does not suppress pixels.
# It only strengthens high-confidence local evidence for image-level AUROC.

def _compute_reliability_map_for_image_topk(
    patch_features: torch.Tensor,
    epoch_text_feature: torch.Tensor,
    img_size: int,
    tau: float = 0.5,
    logit_scale: float = 100.0,
):
    """Return reliability map (B,1,img_size,img_size) from |abnormal-normal| margin."""
    text_feature = epoch_text_feature
    if text_feature.dim() == 3:
        if 1 in text_feature.shape:
            text_feature = text_feature.squeeze()

    if text_feature.dim() == 2:
        scores = 100.0 * torch.einsum("bnc,cd->bnd", patch_features, text_feature)
    else:
        scores = 100.0 * torch.matmul(patch_features, text_feature)

    B, L, C = scores.shape
    if C != 2:
        return torch.ones((B, 1, img_size, img_size), device=patch_features.device, dtype=patch_features.dtype)

    H = int(np.sqrt(L))
    if H * H != L:
        return torch.ones((B, 1, img_size, img_size), device=patch_features.device, dtype=patch_features.dtype)

    tau = max(float(tau), 1e-6)
    logit_scale = max(float(logit_scale), 1e-6)
    margin = (scores[:, :, 1] - scores[:, :, 0]).view(B, H, H) / logit_scale
    reliability = torch.tanh(torch.abs(margin) / tau).unsqueeze(1)  # (B,1,H,H), in [0,1]
    reliability = F.interpolate(reliability, size=img_size, mode="bilinear", align_corners=True)
    return reliability


# def get_predictions(
#     model,
#     class_text_embeddings,
#     test_loader,
#     device,
#     img_size,
#     dataset="MVTec",
# ):
#     """
#     最終版 get_predictions（修正版，支援 dataloader 回傳 dict）:
#     - pixel branch: 使用 multi-scale similarity map + ms_fusion 得到 fused pixel map
#     - image branch: 使用 CLIP-style cosine + softmax 的 anomaly 機率當 image score

#     假設 test_loader 每個 batch 回傳：
#         input_data["image"]     : (B, C, H, W)
#         input_data["mask"]      : (B, 1, H, W)  或 (B, H, W)
#         input_data["label"]     : (B,)
#         input_data["file_name"] : list[str]
#         input_data["class_name"]: list[str]（同一 batch 只會有一個 class）

#     假設 AdaptedCLIP 的 forward 介面為：
#         patch_features, det_feature = model(image)
#         其中 patch_features: List[Tensor(B, L, C)]
#              det_feature   : Tensor(B, C)
#     """
#     model.eval()

#     preds_pixel = []   # list of (B, H, W)
#     preds_image = []   # list of (B,)
#     labels_pixel = []  # list of (B, 1, H, W)
#     labels_image = []  # list of (B,)
#     file_names_all = []

#     domain = DOMAINS[dataset]

#     with torch.no_grad():
#         # ⭐ 關鍵：從 dict 拿資料，而不是 tuple unpack
#         for input_data in tqdm(test_loader):
#             # -------------------------------------------------
#             # 1. 搬到 device
#             # -------------------------------------------------
#             image = input_data["image"].to(device, non_blocking=True)   # (B,C,H,W)
#             mask  = input_data["mask"].to(device, non_blocking=True)    # (B,1,H,W) or (B,H,W)
#             label = input_data["label"].to(device, non_blocking=True)   # (B,)

#             file_name  = input_data["file_name"]                         # list[str]
#             class_name = input_data["class_name"]                        # list[str]
#             # 這裡假設一個 batch 只有單一 class
#             assert len(set(class_name)) == 1, "mixed class not supported in one batch"

#             # 確保 mask shape = (B,1,H,W)
#             if mask.dim() == 3:            # (B,H,W)
#                 mask = mask.unsqueeze(1)   # → (B,1,H,W)

#             B = image.size(0)

#             # -------------------------------------------------
#             # 2. text feature（這個 class 的 text embedding）
#             #    class_text_embeddings: (C, 2) (normal, abnormal)
#             # -------------------------------------------------
#             epoch_text_feature = class_text_embeddings.to(device)  # (C,2)

#             # -------------------------------------------------
#             # 3. forward image → multi-scale patch features + det_feature
#             #    這裡直接呼叫 model(image)，不再使用 return_features 參數
#             # -------------------------------------------------
#             patch_features, det_feature = model(image)   # ✅ 修正點

#             # -------------------------------------------------
#             # 4. pixel branch: multi-scale similarity → ms_fusion
#             # -------------------------------------------------
#             scale_maps = []  # 每個元素: (B, 1, H, W) (只取 abnormal channel)
#             for pf in patch_features:
#                 # pf: (B, L, C)
#                 pixel_map_2ch = calculate_similarity_map(
#                     patch_features=pf,
#                     epoch_text_feature=epoch_text_feature,
#                     img_size=img_size,
#                     test=True,
#                     domain=domain,
#                 )  # (B, 2, H, W)

#                 # 只取 abnormal 通道（index = 1）
#                 scale_maps.append(pixel_map_2ch[:, 1:2, :, :])  # (B,1,H,W)

#             # (B, S, H, W)
#             patch_stack = torch.cat(scale_maps, dim=1)

#             # multi-scale fusion → (B,1,H,W)
#             fused_pixel = model.ms_fusion(patch_stack)   # (B,1,H,W)
#             fused_pixel = fused_pixel.squeeze(1)         # (B,H,W)

#             preds_pixel.append(fused_pixel.cpu().numpy())
#             labels_pixel.append(mask.cpu().numpy())
#             labels_image.append(label.cpu().numpy())
#             file_names_all.extend(file_name)

#             # -------------------------------------------------
#             # 5. image branch: CLIP-style cosine + softmax
#             # -------------------------------------------------
#             epoch_text_feature_norm = F.normalize(epoch_text_feature, dim=0)  # (C,2)
#             det_feature_norm = F.normalize(det_feature, dim=-1)              # (B,C)

#             # (B,2) = [score_normal, score_abnormal]
#             logits_2d = torch.matmul(det_feature_norm, epoch_text_feature_norm)

#             # 使用 P(abnormal) 當 image-level score，對 AUROC 較穩定
#             probs_2d = torch.softmax(logits_2d, dim=1)  # (B,2)
#             image_score = probs_2d[:, 1]                # P(abnormal)

#             preds_image.append(image_score.cpu().numpy())

#     # ---------------------------------------------------------
#     # 6. 整理輸出（與 metrics_eval 介面對齊）
#     # ---------------------------------------------------------
#     pixel_label = np.concatenate(labels_pixel, axis=0)   # (N,1,H,W)
#     image_label = np.concatenate(labels_image, axis=0)   # (N,)
#     pixel_preds = np.concatenate(preds_pixel, axis=0)    # (N,H,W)
#     image_preds = np.concatenate(preds_image, axis=0)    # (N,)

#     return pixel_label, image_label, pixel_preds, image_preds, file_names_all

def _sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _rankdata_np(x: np.ndarray) -> np.ndarray:
    """Small scipy-free rankdata replacement used for Spearman correlation."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def _safe_spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 3 or b.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    ra = _rankdata_np(a)
    rb = _rankdata_np(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    corr = np.corrcoef(ra, rb)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _score_norm_np(scores: np.ndarray, mode: str = "robust_z") -> np.ndarray:
    """Label-free score normalization over the current unlabeled target set."""
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    mode = str(mode).lower()
    if x.size == 0:
        return x.astype(np.float32)
    if mode == "none":
        return x.astype(np.float32)
    if mode == "percentile":
        p10, p50, p90 = np.percentile(x, [10, 50, 90])
        denom = max(float(p90 - p10), 1e-6)
        return ((x - p50) / denom).astype(np.float32)
    if mode == "robust_z":
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        denom = max(1.4826 * mad, 1e-6)
        return ((x - med) / denom).astype(np.float32)
    raise ValueError(f"Unknown --score_norm: {mode}")


def _compute_dense_score_components(pixel_maps: np.ndarray, topk_ratio: float = 0.01):
    """Pass-1 unlabeled dense-score statistics from the map used for image scoring."""
    maps = np.asarray(pixel_maps, dtype=np.float32)
    if maps.ndim == 4:
        flat = maps.reshape(maps.shape[0], -1)
    elif maps.ndim == 3:
        flat = maps.reshape(maps.shape[0], -1)
    else:
        flat = maps.reshape(len(maps), -1)
    n, num_pixels = flat.shape
    if num_pixels == 0:
        z = np.zeros(n, dtype=np.float32)
        return z, z, z
    k = max(1, int(num_pixels * float(topk_ratio)))
    k = min(k, num_pixels)
    topk_idx = np.argpartition(flat, -k, axis=1)[:, -k:]
    topk_vals = np.take_along_axis(flat, topk_idx, axis=1)
    dense_score = topk_vals.mean(axis=1).astype(np.float32)
    med = np.median(flat, axis=1).astype(np.float32)
    mad = np.median(np.abs(flat - med[:, None]), axis=1).astype(np.float32)
    mad = np.maximum(mad, 1e-6)
    concentration = ((dense_score - med) / mad).astype(np.float32)
    high_thr = med + 2.0 * mad
    area_ratio = (flat > high_thr[:, None]).mean(axis=1).astype(np.float32)
    return dense_score, concentration, area_ratio


def _binary_image_metrics(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.0, 0.0
    try:
        auc = round(float(roc_auc_score(y_true, y_score)), 4) * 100.0
        ap = round(float(average_precision_score(y_true, y_score)), 4) * 100.0
        return auc, ap
    except Exception:
        return 0.0, 0.0


def _fit_target_adaptive_scores(records, args, logger=None, epoch=0):
    """V27b Pass-2/3 label-free target score calibration.

    Design notes:
    - Calibration uses predictions only, never labels.
    - Default is dense-first and safe: alpha stays in [0.90, 1.00].
    - Per-class calibration is default because VisA / MVTec classes have very different score scales.
    - CLS can only make a small correction, and only when dense and CLS scores are positively correlated
      on the unlabeled target distribution.
    """
    alpha_min = float(args.alpha_min)
    alpha_max = float(args.alpha_max)
    if alpha_max < alpha_min:
        alpha_min, alpha_max = alpha_max, alpha_min
    tau = max(float(args.adaptive_alpha_tau), 1e-6)
    center = float(args.adaptive_alpha_center)
    mode = str(args.image_score_mode).lower()
    target_mode = str(args.target_calib_mode).lower()
    calib_scope = str(getattr(args, "calib_scope", "per_class")).lower()
    safe_cls_gate = bool(getattr(args, "safe_cls_gate", True))
    corr_thr = float(getattr(args, "safe_cls_corr_threshold", 0.15))
    corr_width = max(float(getattr(args, "safe_cls_corr_width", 0.10)), 1e-6)

    if target_mode not in ["none", "unlabeled"]:
        raise ValueError(f"Unknown --target_calib_mode: {target_mode}")
    if calib_scope not in ["global", "per_class"]:
        raise ValueError(f"Unknown --calib_scope: {calib_scope}")

    def _calibrate_one(dense, cls, concentration, area_ratio):
        dense = np.asarray(dense, dtype=np.float32).reshape(-1)
        cls = np.asarray(cls, dtype=np.float32).reshape(-1)
        concentration = np.asarray(concentration, dtype=np.float32).reshape(-1)
        area_ratio = np.asarray(area_ratio, dtype=np.float32).reshape(-1)

        dense_z = _score_norm_np(dense, args.score_norm)
        cls_z = _score_norm_np(cls, args.score_norm)
        corr = _safe_spearman_np(dense_z, cls_z)
        conc_med = float(np.median(concentration)) if concentration.size else 0.0
        area_med = float(np.median(area_ratio)) if area_ratio.size else 0.0
        conc_rel = float(_sigmoid_np((conc_med - center) / tau))

        # Dense-first policy:
        #   alpha_max means dense-only.  Lower alpha means allowing CLS to correct.
        #   We lower alpha only when the map looks less concentrated AND CLS agrees with dense.
        if target_mode == "none":
            alpha_dataset = alpha_max
            dataset_reliability = 1.0
            cls_mix_budget = 0.0
        else:
            if safe_cls_gate and corr < corr_thr:
                cls_mix_budget = 0.0
            else:
                corr_rel = float(_sigmoid_np((corr - corr_thr) / corr_width))
                # High concentration => dense is reliable => smaller CLS budget.
                cls_mix_budget = float(np.clip((1.0 - conc_rel) * corr_rel, 0.0, 1.0))
            alpha_dataset = alpha_max - (alpha_max - alpha_min) * cls_mix_budget
            alpha_dataset = float(np.clip(alpha_dataset, alpha_min, alpha_max))
            dataset_reliability = 1.0 - cls_mix_budget

        # Per-image safety: if a map is concentrated, move back to dense-only.
        per_image_rel = _sigmoid_np((concentration - center) / tau).astype(np.float32)
        alpha_image = alpha_dataset + (alpha_max - alpha_dataset) * per_image_rel
        alpha_image = np.clip(alpha_image, alpha_min, alpha_max).astype(np.float32)

        if mode == "adaptive_hybrid":
            final_score = alpha_image * dense_z + (1.0 - alpha_image) * cls_z
        elif mode == "cls":
            final_score = cls_z
            alpha_image = np.zeros_like(alpha_image, dtype=np.float32)
        elif mode == "dense":
            final_score = dense_z
            alpha_image = np.ones_like(alpha_image, dtype=np.float32)
        else:
            raise ValueError(f"_fit_target_adaptive_scores called with unsupported mode: {mode}")

        return {
            "dense_z": dense_z.astype(np.float32),
            "cls_z": cls_z.astype(np.float32),
            "final_score": final_score.astype(np.float32),
            "alpha": alpha_image.astype(np.float32),
            "corr": float(corr),
            "conc_med": float(conc_med),
            "area_med": float(area_med),
            "alpha_dataset": float(alpha_dataset),
            "alpha_mean": float(np.mean(alpha_image)) if alpha_image.size else 0.0,
            "alpha_std": float(np.std(alpha_image)) if alpha_image.size else 0.0,
            "dataset_reliability": float(dataset_reliability),
            "cls_mix_budget": float(cls_mix_budget),
        }

    out = {}
    summaries = []

    if calib_scope == "global":
        dense_all = np.concatenate([r["dense_score"] for r in records], axis=0).astype(np.float32)
        cls_all = np.concatenate([r["cls_score"] for r in records], axis=0).astype(np.float32)
        conc_all = np.concatenate([r["concentration"] for r in records], axis=0).astype(np.float32)
        area_all = np.concatenate([r["area_ratio"] for r in records], axis=0).astype(np.float32)
        res = _calibrate_one(dense_all, cls_all, conc_all, area_all)
        offset = 0
        for r in records:
            n = int(r["n"])
            out[r["class_name"]] = {
                "final_score": res["final_score"][offset:offset+n].astype(np.float32),
                "dense_z": res["dense_z"][offset:offset+n].astype(np.float32),
                "cls_z": res["cls_z"][offset:offset+n].astype(np.float32),
                "alpha": res["alpha"][offset:offset+n].astype(np.float32),
                "dense_score": r["dense_score"].astype(np.float32),
                "cls_score": r["cls_score"].astype(np.float32),
                "concentration": r["concentration"].astype(np.float32),
                "area_ratio": r["area_ratio"].astype(np.float32),
                "cache_file": r["cache_file"],
                "calib_summary": res,
            }
            offset += n
        summaries.append(res)
    else:
        for r in records:
            res = _calibrate_one(r["dense_score"], r["cls_score"], r["concentration"], r["area_ratio"])
            out[r["class_name"]] = {
                "final_score": res["final_score"].astype(np.float32),
                "dense_z": res["dense_z"].astype(np.float32),
                "cls_z": res["cls_z"].astype(np.float32),
                "alpha": res["alpha"].astype(np.float32),
                "dense_score": r["dense_score"].astype(np.float32),
                "cls_score": r["cls_score"].astype(np.float32),
                "concentration": r["concentration"].astype(np.float32),
                "area_ratio": r["area_ratio"].astype(np.float32),
                "cache_file": r["cache_file"],
                "calib_summary": res,
            }
            summaries.append(res)
            if logger is not None:
                logger.info(
                    "V27b per-class calibration epoch=%s class=%s mode=%s target_calib=%s score_norm=%s "
                    "alpha_dataset=%.4f alpha_mean=%.4f alpha_std=%.4f dense_cls_spearman=%.4f "
                    "conc_median=%.4f area_median=%.6f cls_mix_budget=%.4f",
                    epoch, r["class_name"], mode, target_mode, args.score_norm,
                    res["alpha_dataset"], res["alpha_mean"], res["alpha_std"], res["corr"],
                    res["conc_med"], res["area_med"], res["cls_mix_budget"],
                )

    def _mean_key(key):
        vals = [s[key] for s in summaries]
        return float(np.mean(vals)) if len(vals) else 0.0

    summary = {
        "alpha_dataset": _mean_key("alpha_dataset"),
        "alpha_mean": _mean_key("alpha_mean"),
        "alpha_std": _mean_key("alpha_std"),
        "dense_cls_spearman": _mean_key("corr"),
        "concentration_median": _mean_key("conc_med"),
        "area_ratio_median": _mean_key("area_med"),
        "dataset_reliability": _mean_key("dataset_reliability"),
        "cls_mix_budget": _mean_key("cls_mix_budget"),
        "calib_scope": calib_scope,
    }
    if logger is not None:
        logger.info(
            "V27b target adaptive summary epoch=%s scope=%s mode=%s target_calib=%s score_norm=%s "
            "alpha_mean=%.4f alpha_std=%.4f dense_cls_spearman_mean=%.4f conc_median_mean=%.4f "
            "cls_mix_budget_mean=%.4f safe_cls_gate=%s",
            epoch, calib_scope, mode, target_mode, args.score_norm,
            summary["alpha_mean"], summary["alpha_std"], summary["dense_cls_spearman"],
            summary["concentration_median"], summary["cls_mix_budget"], str(safe_cls_gate),
        )
    return out, summary

def get_predictions(
    model,
    class_text_embeddings,
    test_loader,
    device,
    img_size,
    dataset="MVTec",
    fb_bg_suppression_beta: float = 0.0,
    fb_bg_suppression_mode: str = "normal_z",
    semantic_uncertainty_gamma: float = 0.0,
    semantic_uncertainty_tau: float = 0.5,
    semantic_uncertainty_logit_scale: float = 100.0,
    use_reliable_topk: bool = False,
    reliable_topk_rho: float = 0.0,
    reliable_topk_tau: float = 0.5,
    reliable_topk_logit_scale: float = 100.0,
    use_score_calibrator: bool = False,
    topk_ratio: float = 0.01,
):
    """
    穩定版 get_predictions：

    - pixel branch:
        對每個 multi-scale patch_features 呼叫 calculate_similarity_map(..., test=True)，
        其回傳 (B,1,H,W) 單通道 anomaly map，最後對多尺度做 channel-wise mean fusion。

    - image branch:
        使用 CLIP-style cosine + softmax P(abnormal) 當 image score，
        後續再交給 metrics_eval 做 top-k reweight 來提升 image AUROC。

    V11:
        FB-CLIP-style background suppression is applied only to pixel_preds.
        A raw V5 pixel map is also returned for image-level top-k fusion so that
        the image branch remains comparable to V5.

    V11b:
        Semantic-uncertainty suppression additionally down-weights pixel responses
        where normal/abnormal anchors are too close. It does not modify the raw V5
        pixel map used for image-level top-k fusion.

    V22:
        Image-aware reliable top-k fusion.  It keeps pixel_preds unchanged, but
        replaces pixel_preds_for_image with a reliability-weighted raw map:
        raw * (1 + rho * tanh(abs(abnormal-normal)/logit_scale/tau)).
    """
    model.eval()

    preds_pixel = []   # list of (B, H, W), possibly background-suppressed V11 map
    preds_pixel_for_image = []  # list of (B, H, W), raw V5 map for image top-k fusion
    preds_image = []   # list of (B,)
    labels_pixel = []  # list of (B, 1, H, W)
    labels_image = []  # list of (B,)
    file_names_all = []

    domain = DOMAINS[dataset]

    with torch.no_grad():
        for input_data in tqdm(test_loader):
            # -------------------------------------------------
            # 1. 搬到 device
            # -------------------------------------------------
            image = input_data["image"].to(device, non_blocking=True)   # (B,C,H,W)
            mask  = input_data["mask"].to(device, non_blocking=True)    # (B,1,H,W) or (B,H,W)
            label = input_data["label"].to(device, non_blocking=True)   # (B,)

            file_name  = input_data["file_name"]   # list[str]
            class_name = input_data["class_name"]  # list[str]
            assert len(set(class_name)) == 1, "mixed class not supported in one batch"

            # 確保 mask shape = (B,1,H,W)
            if mask.dim() == 3:            # (B,H,W)
                mask = mask.unsqueeze(1)   # → (B,1,H,W)

            B = image.size(0)

            # -------------------------------------------------
            # 2. text feature（這個 class 的 text embedding）
            #    class_text_embeddings: (C, 2) (normal, abnormal)
            # -------------------------------------------------
            epoch_text_feature = class_text_embeddings.to(device)  # (C,2)

            # -------------------------------------------------
            # 3. forward image → multi-scale patch features + det_feature
            # -------------------------------------------------
            patch_features, det_feature = model(image)  # 不再用 return_features=True

            # 有些實作可能回傳 Tensor 而不是 list，統一包成 list
            if isinstance(patch_features, torch.Tensor):
                patch_features_list = [patch_features]
            else:
                patch_features_list = list(patch_features)

            # -------------------------------------------------
            # 4. pixel branch: multi-scale similarity → simple mean fusion
            #    calculate_similarity_map(test=True) 回傳 (B,1,H,W) 單通道 anomaly map
            # -------------------------------------------------
            scale_maps = []      # V11 pixel map, possibly background-suppressed; each: (B,1,H,W)
            raw_scale_maps = []  # raw V5 map for image-level top-k fusion; each: (B,1,H,W)
            reliable_scale_maps = []  # V22 reliable raw maps for image-level top-k fusion

            for pf in patch_features_list:
                pixel_map, raw_pixel_map = calculate_similarity_map(
                    patch_features=pf,                  # (B, N, C_feat)
                    epoch_text_feature=epoch_text_feature,  # (C,2)
                    img_size=img_size,
                    test=True,
                    domain=domain,
                    fb_bg_suppression_beta=fb_bg_suppression_beta,
                    fb_bg_suppression_mode=fb_bg_suppression_mode,
                    semantic_uncertainty_gamma=semantic_uncertainty_gamma,
                    semantic_uncertainty_tau=semantic_uncertainty_tau,
                    semantic_uncertainty_logit_scale=semantic_uncertainty_logit_scale,
                    return_raw_map=True,
                )  # pixel_map/raw_pixel_map shape = (B,1,H,W)

                scale_maps.append(pixel_map)
                raw_scale_maps.append(raw_pixel_map)

                # V22: only change image-level top-k evidence, not pixel metrics.
                if bool(use_reliable_topk) and float(reliable_topk_rho) > 0:
                    reliability = _compute_reliability_map_for_image_topk(
                        patch_features=pf,
                        epoch_text_feature=epoch_text_feature,
                        img_size=img_size,
                        tau=reliable_topk_tau,
                        logit_scale=reliable_topk_logit_scale,
                    )
                    reliable_raw_pixel_map = raw_pixel_map * (1.0 + float(reliable_topk_rho) * reliability)
                else:
                    reliable_raw_pixel_map = raw_pixel_map
                reliable_scale_maps.append(reliable_raw_pixel_map)

            if len(scale_maps) == 0:
                # 理論上不會發生，保險起見給 0 map
                fused_pixel = torch.zeros(
                    (B, img_size, img_size), device=device, dtype=torch.float32
                )
                raw_fused_pixel = fused_pixel
                reliable_fused_pixel = fused_pixel
            elif len(scale_maps) == 1:
                fused_pixel = scale_maps[0].squeeze(1)
                raw_fused_pixel = raw_scale_maps[0].squeeze(1)
                reliable_fused_pixel = reliable_scale_maps[0].squeeze(1)
            else:
                # scale_maps: list of (B,1,H,W)
                patch_stack = torch.cat(scale_maps, dim=1)       # (B,S,H,W)
                raw_patch_stack = torch.cat(raw_scale_maps, dim=1)
                reliable_patch_stack = torch.cat(reliable_scale_maps, dim=1)
                fused_pixel = model.ms_fusion(patch_stack).squeeze(1)               # (B,H,W)
                raw_fused_pixel = model.ms_fusion(raw_patch_stack).squeeze(1)       # (B,H,W)
                reliable_fused_pixel = model.ms_fusion(reliable_patch_stack).squeeze(1)  # (B,H,W)

            preds_pixel.append(fused_pixel.cpu().numpy())
            # V22: pass reliability-weighted raw map only to image-level top-k fusion.
            preds_pixel_for_image.append(reliable_fused_pixel.cpu().numpy())
            labels_pixel.append(mask.cpu().numpy())
            labels_image.append(label.cpu().numpy())
            file_names_all.extend(file_name)

            # -------------------------------------------------
            # 5. image branch: CLIP-style cosine + softmax P(abnormal)
            # -------------------------------------------------
            epoch_text_feature_norm = F.normalize(epoch_text_feature, dim=0)  # (C,2)
            det_feature_norm = F.normalize(det_feature, dim=-1)              # (B,C_feat)

            logits_2d = torch.matmul(det_feature_norm, epoch_text_feature_norm)  # (B,2)
            probs_2d = torch.softmax(logits_2d, dim=1)  # (B,2)
            image_score = probs_2d[:, 1]                # P(abnormal)

            # V28: optional source-trained dense-to-image score calibrator.
            # It outputs final image score directly. metrics_eval should then be called
            # with image_fusion_alpha=0.0 so it uses this score rather than re-blending.
            if bool(use_score_calibrator) and getattr(model, "score_calibrator", None) is not None:
                image_score, alpha_i, dense_score, _ = model.calibrated_image_score(
                    reliable_fused_pixel,
                    image_score,
                    topk_ratio=topk_ratio,
                )

            preds_image.append(image_score.cpu().numpy())

    # ---------------------------------------------------------
    # 6. 整理輸出（與 metrics_eval 介面對齊）
    # ---------------------------------------------------------
    pixel_label = np.concatenate(labels_pixel, axis=0)   # (N,1,H,W)
    image_label = np.concatenate(labels_image, axis=0)   # (N,)
    pixel_preds = np.concatenate(preds_pixel, axis=0)    # (N,H,W)
    pixel_preds_for_image = np.concatenate(preds_pixel_for_image, axis=0)  # (N,H,W)
    image_preds = np.concatenate(preds_image, axis=0)    # (N,)

    return pixel_label, image_label, pixel_preds, image_preds, pixel_preds_for_image, file_names_all


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-B-16-plus-240, ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    # testing
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--data_path", type=str, default=None, help="Optional dataset root override, e.g. D:/data/mvtec_anomaly_detection")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers; use 0 for easier path debugging")
    parser.add_argument("--eval_epochs", type=str, default=None, help="Comma-separated epochs to evaluate, e.g. 4,6,8,13,15. Default: all valid checkpoints. Use 'best_source' or --eval_best_source for V28 best-source checkpoint.")
    parser.add_argument("--eval_best_source", action="store_true", help="evaluate image_adapter_best_source.pth saved by V28 source-only selection")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=6)
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)

    # Optional frozen DINOv3 dense-token branch. Use the same flags as training.
    parser.add_argument("--use_dinov3", action="store_true", help="enable frozen DINOv3 dense-token fusion in the image branch")
    parser.add_argument("--dino_model_name", type=str, default="facebook/dinov3-vith16plus-pretrain-lvd1689m",
                        help="Hugging Face DINOv3 backbone name or local path")
    parser.add_argument("--dino_fusion_alpha", type=float, default=0.05,
                        help="residual DINOv3 -> CLIP token fusion strength; must match training for evaluation")
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

    parser.add_argument("--disable_text_adapter", action="store_true", help="run ablation without text adapter")
    parser.add_argument("--disable_image_adapter", action="store_true", help="run ablation without image adapter")

    # V11: FB-CLIP-style pixel-only background suppression.
    # beta=0.0 exactly recovers V5 behavior.
    parser.add_argument("--fb_bg_suppression_beta", type=float, default=0.0,
                        help="V11 pixel-only background suppression strength; 0.0 keeps original V5")
    parser.add_argument("--fb_bg_suppression_mode", type=str, default="normal_z",
                        choices=["normal_z", "normal_logit", "margin_gate"],
                        help="background suppression mode for pixel map")

    # V11b: semantic-uncertainty suppression on pixel branch only.
    # gamma=0.0 exactly recovers V11 behavior.
    parser.add_argument("--semantic_uncertainty_gamma", type=float, default=0.0,
                        help="V11b uncertainty suppression strength; 0.0 keeps original V11")
    parser.add_argument("--semantic_uncertainty_tau", type=float, default=0.5,
                        help="temperature for exp(-abs(margin)/tau) uncertainty gate")
    parser.add_argument("--semantic_uncertainty_logit_scale", type=float, default=100.0,
                        help="normalizes CLIP logits before uncertainty gate; keep 100.0 for this codebase")

    parser.add_argument("--image_fusion_alpha", type=float, default=0.7,
                        help="V5 image-level top-k fusion alpha; keep 0.7 for comparability")
    parser.add_argument("--topk_ratio", type=float, default=0.02,
                        help="top-k pixel ratio used for image-level fusion")

    # V22: image-aware reliable top-k fusion.  This only changes the map used by
    # image-level top-k fusion; pixel metrics still use the original V13+V14/V11 map.
    parser.add_argument("--use_reliable_topk", action="store_true",
                        help="enable V22 reliability-weighted raw pixel map for image-level top-k fusion")
    parser.add_argument("--reliable_topk_rho", type=float, default=0.0,
                        help="V22 strength. Suggested: 0.03 or 0.05. 0.0 disables it")
    parser.add_argument("--reliable_topk_tau", type=float, default=0.5,
                        help="V22 tanh margin temperature")
    parser.add_argument("--reliable_topk_logit_scale", type=float, default=100.0,
                        help="CLIP logit scale used when converting margin to cosine scale")

    # V28: source-trained dense-to-image score calibrator.
    parser.add_argument("--use_score_calibrator", action="store_true",
                        help="use score_calibrator stored in V28 checkpoint for image-level score")
    parser.add_argument("--score_calib_alpha_min", type=float, default=0.85,
                        help="minimum dense-score weight used to initialize the V28 score calibrator")
    parser.add_argument("--score_calib_alpha_max", type=float, default=1.0,
                        help="maximum dense-score weight used to initialize the V28 score calibrator")

    # V27: label-free target-adaptive image-score calibration.
    parser.add_argument("--image_score_mode", type=str, default="fixed_hybrid",
                        choices=["fixed_hybrid", "adaptive_hybrid", "cls", "dense"],
                        help="fixed_hybrid keeps original metrics_eval fusion; adaptive_hybrid enables pass1/2/3 target calibration")
    parser.add_argument("--target_calib_mode", type=str, default="none",
                        choices=["none", "unlabeled"],
                        help="unlabeled uses target prediction distribution only; no labels are used for calibration")
    parser.add_argument("--score_norm", type=str, default="robust_z",
                        choices=["none", "robust_z", "percentile"],
                        help="normalization for dense/CLS scores before adaptive fusion")
    parser.add_argument("--alpha_min", type=float, default=0.85,
                        help="V27b minimum per-image dense-score weight for adaptive_hybrid; safe default updated to 0.85 after MVTec->VisA alpha sweep")
    parser.add_argument("--alpha_max", type=float, default=1.0,
                        help="maximum per-image dense-score weight for adaptive_hybrid")
    parser.add_argument("--adaptive_alpha_tau", type=float, default=0.5,
                        help="temperature for map-concentration reliability -> alpha")
    parser.add_argument("--adaptive_alpha_center", type=float, default=1.0,
                        help="center for map-concentration reliability -> alpha")
    parser.add_argument("--log_score_components", action="store_true",
                        help="add cls/dense/final image-score component metrics to the result table")
    parser.add_argument("--save_score_components", action="store_true",
                        help="save per-image score components CSV for debugging/analysis")
    parser.add_argument("--keep_calib_cache", action="store_true",
                        help="keep temporary NPZ cache files created by adaptive_hybrid pass1/2/3")

    args = parser.parse_args()
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "test.log"),
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.info("args: %s", vars(args))
    _apply_data_path_override(args, logger)
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model
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

    # ---------- 1) 只載入一次 text adapter ----------
    text_file = glob(args.save_path + "/text_adapter.pth")
    if (not args.disable_text_adapter) and len(text_file) > 0:
        checkpoint = torch.load(text_file[0], map_location=device)
        # model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])

        if "text_layer_gates" in checkpoint:
            model.text_layer_gates.data.copy_(checkpoint["text_layer_gates"].to(device))
            
        
        adapt_text = True
    else:
        adapt_text = False

    # ---------- 2) 只建一次 dataset + dataloader ----------
    kwargs = {"num_workers": int(args.num_workers), "pin_memory": True} if use_cuda else {"num_workers": int(args.num_workers)}
    image_datasets = get_dataset(
        args.dataset,
        args.img_size,
        None,
        args.shot,
        "test",
        logger=logger,
    )
    dataloaders = {
        class_name: torch.utils.data.DataLoader(
            image_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs,
        )
        for class_name, image_dataset in image_datasets.items()
    }

    # ---------- 3) 只算一次「所有 class 的 text embeddings」 ----------
    # with torch.no_grad():
    with torch.inference_mode():
        if adapt_text:
            text_embeddings = get_adapted_text_embedding(
                model, args.dataset, device
            )
        else:
            text_embeddings = get_adapted_text_embedding(
                clip_model, args.dataset, device
            )

    # ---------- 4) 對每個 image_adapter checkpoint 重複跑 inference ----------
    # files = sorted(glob(args.save_path + "/image_adapter_*.pth"))
    # files = files[-5:]  # 例如只測最後 5 個
    # assert len(files) > 0, "image adapter checkpoint not found"

    # for file in files:
    #     checkpoint = torch.load(file, map_location=device)
    #     model.image_adapter.load_state_dict(checkpoint["image_adapter"])
    #     test_epoch = checkpoint["epoch"]
    #     logger.info("-----------------------------------------------")
    #     logger.info("load model from epoch %d", test_epoch)
    #     logger.info("-----------------------------------------------")

    def extract_epoch(path: str) -> int:
        """Extract epoch from supported checkpoint names.

        Supported:
          image_adapter_17.pth
          image_adapter_ema_17.pth
        Explicitly excludes image_adapter_raw_17.pth and image_adapter.pth.
        """
        base = os.path.basename(path)
        m = re.match(r"^image_adapter_(\d+)\.pth$", base)
        if m:
            return int(m.group(1))
        m = re.match(r"^image_adapter_ema_(\d+)\.pth$", base)
        if m:
            return int(m.group(1))
        return -1

    def parse_eval_epochs(value):
        if value is None or str(value).strip() == "":
            return None
        if str(value).strip().lower() in ["best", "best_source", "source_best"]:
            args.eval_best_source = True
            return None
        out = []
        for item in str(value).split(','):
            item = item.strip()
            if item:
                out.append(int(item))
        return set(out)

    if args.disable_image_adapter:
        files = [None]
        print("Will evaluate image-only disabled baseline (no image adapter checkpoint).")
    else:
        # V13+V14 saves EMA as image_adapter_{epoch}.pth when --save_ema_as_eval is used,
        # and raw weights as image_adapter_raw_{epoch}.pth.  The old glob matched raw
        # checkpoints too, producing many epoch=-1 entries.  Keep only valid eval ckpts.
        all_ckpts = glob(os.path.join(args.save_path, "image_adapter_*.pth"))
        ckpt_paths = [p for p in all_ckpts if extract_epoch(p) > 0]

        # If a future version saves explicit image_adapter_ema_{epoch}.pth files and no
        # standard image_adapter_{epoch}.pth files, the regex above still supports them.
        ckpt_paths = sorted(ckpt_paths, key=extract_epoch)

        wanted_epochs = parse_eval_epochs(args.eval_epochs)
        if bool(args.eval_best_source):
            best_path = os.path.join(args.save_path, "image_adapter_best_source.pth")
            if not os.path.isfile(best_path):
                raise FileNotFoundError(f"--eval_best_source requested but not found: {best_path}")
            ckpt_paths = [best_path]
            wanted_epochs = None

        if wanted_epochs is not None:
            ckpt_paths = [p for p in ckpt_paths if extract_epoch(p) in wanted_epochs]

        if len(ckpt_paths) == 0:
            raise FileNotFoundError(
                "No valid image adapter checkpoints found. Expected files like "
                "image_adapter_1.pth ... image_adapter_20.pth in "
                f"{args.save_path}. Raw checkpoints image_adapter_raw_*.pth are ignored."
            )

        files = ckpt_paths
        print("Will evaluate image_adapter epochs:", [extract_epoch(f) for f in files])

    for file in files:
        if file is not None:
            checkpoint = torch.load(file, map_location=device)
            image_state = checkpoint["image_adapter"]
            ckpt_has_dino = any(str(k).startswith("dinov3_proj") for k in image_state.keys())
            if ckpt_has_dino and (not args.use_dinov3):
                raise ValueError("This image_adapter checkpoint contains DINOv3 projection weights. Re-run test.py with --use_dinov3 and the same DINOv3 settings used for training.")
            if args.use_dinov3 and (not ckpt_has_dino):
                raise ValueError("--use_dinov3 was enabled, but this image_adapter checkpoint has no DINOv3 projection weights. Train the image adapter with --use_dinov3 first, or evaluate without --use_dinov3.")
            model.image_adapter.load_state_dict(image_state)

            if "ms_fusion" in checkpoint:
                model.ms_fusion.load_state_dict(checkpoint["ms_fusion"])

            if "image_layer_gates" in checkpoint:
                model.image_layer_gates.data.copy_(checkpoint["image_layer_gates"].to(device))
            if getattr(model, "score_calibrator", None) is not None:
                if "score_calibrator" not in checkpoint:
                    raise ValueError("--use_score_calibrator was enabled, but this checkpoint has no score_calibrator. Use a V28 checkpoint or remove --use_score_calibrator.")
                model.score_calibrator.load_state_dict(checkpoint["score_calibrator"])
            test_epoch = checkpoint.get("epoch", extract_epoch(file))
        else:
            test_epoch = 0

        logger.info("-----------------------------------------------")
        logger.info("load model from epoch %d", test_epoch)
        logger.info("-----------------------------------------------")

        # df = DataFrame(
        #     columns=[
        #         "class name",
        #         "pixel AUC",
        #         "pixel AP",
        #         "image AUC",
        #         "image AP",
        #     ]
        # )

        df = DataFrame(
            columns=[
                "class name",
                "pixel AUC",
                "pixel AP",
                "PRO",
                "F1",
                "IoU",
                "image AUC",
                "image AP",
            ]
        )
                

        # ------------------------------------------------------------------
        # V27 mode switch:
        #   fixed_hybrid: original one-pass per-class evaluation.
        #   adaptive_hybrid/cls/dense: Pass 1/2/3 target-adaptive calibration.
        # ------------------------------------------------------------------
        image_score_mode = str(args.image_score_mode).lower()

        if image_score_mode == "fixed_hybrid":
            # Original path: no target-level cache/calibration; exact backward compatibility.
            for class_name, image_dataloader in dataloaders.items():
                with torch.inference_mode():
                    class_text_embeddings = text_embeddings[class_name]
                    masks, labels, preds, preds_image, preds_for_image, file_names = get_predictions(
                        model=model,
                        class_text_embeddings=class_text_embeddings,
                        test_loader=image_dataloader,
                        device=device,
                        img_size=args.img_size,
                        dataset=args.dataset,
                        fb_bg_suppression_beta=args.fb_bg_suppression_beta,
                        fb_bg_suppression_mode=args.fb_bg_suppression_mode,
                        semantic_uncertainty_gamma=args.semantic_uncertainty_gamma,
                        semantic_uncertainty_tau=args.semantic_uncertainty_tau,
                        semantic_uncertainty_logit_scale=args.semantic_uncertainty_logit_scale,
                        use_reliable_topk=args.use_reliable_topk,
                        reliable_topk_rho=args.reliable_topk_rho,
                        reliable_topk_tau=args.reliable_topk_tau,
                        reliable_topk_logit_scale=args.reliable_topk_logit_scale,
                        use_score_calibrator=args.use_score_calibrator,
                        topk_ratio=args.topk_ratio,
                    )

                if args.visualize:
                    visualize(masks, preds, file_names, args.save_path, args.dataset, class_name=class_name)
                class_result_dict = metrics_eval(
                    masks, labels, preds, preds_image, class_name,
                    domain=DOMAINS[args.dataset],
                    pixel_preds_for_image=preds_for_image,
                    image_fusion_alpha=(0.0 if args.use_score_calibrator else args.image_fusion_alpha),
                    topk_ratio=args.topk_ratio,
                )
                df.loc[len(df)] = Series(class_result_dict)
                del masks, labels, preds, preds_image, preds_for_image, file_names
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # =========================
            # Pass 1: collect unlabeled target score components and cache large arrays.
            # =========================
            cache_root = os.path.join(args.save_path, "target_adaptive_cache", f"{args.dataset}_epoch_{test_epoch}")
            if os.path.isdir(cache_root):
                shutil.rmtree(cache_root)
            os.makedirs(cache_root, exist_ok=True)
            logger.info("V27 Pass 1: collecting target predictions and score components into %s", cache_root)
            records = []
            per_image_rows = []
            for class_name, image_dataloader in dataloaders.items():
                with torch.inference_mode():
                    class_text_embeddings = text_embeddings[class_name]
                    masks, labels, preds, preds_image, preds_for_image, file_names = get_predictions(
                        model=model,
                        class_text_embeddings=class_text_embeddings,
                        test_loader=image_dataloader,
                        device=device,
                        img_size=args.img_size,
                        dataset=args.dataset,
                        fb_bg_suppression_beta=args.fb_bg_suppression_beta,
                        fb_bg_suppression_mode=args.fb_bg_suppression_mode,
                        semantic_uncertainty_gamma=args.semantic_uncertainty_gamma,
                        semantic_uncertainty_tau=args.semantic_uncertainty_tau,
                        semantic_uncertainty_logit_scale=args.semantic_uncertainty_logit_scale,
                        use_reliable_topk=args.use_reliable_topk,
                        reliable_topk_rho=args.reliable_topk_rho,
                        reliable_topk_tau=args.reliable_topk_tau,
                        reliable_topk_logit_scale=args.reliable_topk_logit_scale,
                        use_score_calibrator=args.use_score_calibrator,
                        topk_ratio=args.topk_ratio,
                    )
                dense_score, concentration, area_ratio = _compute_dense_score_components(preds_for_image, topk_ratio=args.topk_ratio)
                labels_1d = np.asarray(labels).reshape(-1).astype(np.int64)
                preds_image_1d = np.asarray(preds_image).reshape(-1).astype(np.float32)
                n = int(labels_1d.shape[0])
                cache_file = os.path.join(cache_root, f"{class_name}.npz")
                np.savez(
                    cache_file,
                    masks=np.asarray(masks),
                    labels=labels_1d,
                    preds=np.asarray(preds, dtype=np.float32),
                    preds_for_image=np.asarray(preds_for_image, dtype=np.float32),
                    preds_image=preds_image_1d,
                    file_names=np.asarray(file_names, dtype=object),
                )
                records.append({
                    "class_name": class_name,
                    "n": n,
                    "cache_file": cache_file,
                    "dense_score": dense_score.astype(np.float32),
                    "cls_score": preds_image_1d.astype(np.float32),
                    "concentration": concentration.astype(np.float32),
                    "area_ratio": area_ratio.astype(np.float32),
                })
                if args.save_score_components:
                    for i in range(n):
                        per_image_rows.append({
                            "epoch": test_epoch,
                            "class_name": class_name,
                            "file_name": str(file_names[i]) if i < len(file_names) else "",
                            "label": int(labels_1d[i]),
                            "cls_score": float(preds_image_1d[i]),
                            "dense_score": float(dense_score[i]),
                            "map_concentration": float(concentration[i]),
                            "map_area_ratio": float(area_ratio[i]),
                        })
                del masks, labels, preds, preds_image, preds_for_image, file_names
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # =========================
            # Pass 2: fit label-free target calibration over all target images.
            # =========================
            logger.info("V27 Pass 2: fitting label-free target calibration")
            calibrated, calib_summary = _fit_target_adaptive_scores(records, args, logger=logger, epoch=test_epoch)

            if args.save_score_components and len(per_image_rows) > 0:
                cursor_by_class = {r["class_name"]: 0 for r in records}
                for row in per_image_rows:
                    c = row["class_name"]
                    j = cursor_by_class[c]
                    row["cls_z"] = float(calibrated[c]["cls_z"][j])
                    row["dense_z"] = float(calibrated[c]["dense_z"][j])
                    row["adaptive_alpha"] = float(calibrated[c]["alpha"][j])
                    row["final_score"] = float(calibrated[c]["final_score"][j])
                    cursor_by_class[c] += 1
                comp_csv = os.path.join(args.save_path, f"score_components_{args.dataset}_epoch_{test_epoch}.csv")
                pd.DataFrame(per_image_rows).to_csv(comp_csv, index=False, encoding="utf-8-sig")
                logger.info("saved score component CSV: %s", comp_csv)

            # =========================
            # Pass 3: evaluate per class with calibrated image score.
            # Pixel metrics still use the original pixel map.
            # =========================
            logger.info("V27 Pass 3: evaluating with calibrated image scores")
            for rec in records:
                class_name = rec["class_name"]
                pack = np.load(rec["cache_file"], allow_pickle=True)
                masks = pack["masks"]
                labels = pack["labels"]
                preds = pack["preds"]
                preds_for_image = pack["preds_for_image"]
                file_names = pack["file_names"].tolist()
                final_score = calibrated[class_name]["final_score"]
                if args.visualize:
                    visualize(masks, preds, file_names, args.save_path, args.dataset, class_name=class_name)
                class_result_dict = metrics_eval(
                    masks, labels, preds, final_score, class_name,
                    domain=DOMAINS[args.dataset],
                    pixel_preds_for_image=preds_for_image,
                    image_fusion_alpha=0.0,
                    topk_ratio=args.topk_ratio,
                )
                if args.log_score_components:
                    cls_auc, cls_ap = _binary_image_metrics(labels, calibrated[class_name]["cls_score"])
                    dense_auc, dense_ap = _binary_image_metrics(labels, calibrated[class_name]["dense_score"])
                    cls_z_auc, cls_z_ap = _binary_image_metrics(labels, calibrated[class_name]["cls_z"])
                    dense_z_auc, dense_z_ap = _binary_image_metrics(labels, calibrated[class_name]["dense_z"])
                    fixed07 = 0.7 * calibrated[class_name]["dense_z"] + 0.3 * calibrated[class_name]["cls_z"]
                    fixed07_auc, fixed07_ap = _binary_image_metrics(labels, fixed07)
                    final_auc, final_ap = _binary_image_metrics(labels, calibrated[class_name]["final_score"])
                    class_result_dict.update({
                        "cls AUC": cls_auc,
                        "cls AP": cls_ap,
                        "dense AUC": dense_auc,
                        "dense AP": dense_ap,
                        "cls-z AUC": cls_z_auc,
                        "cls-z AP": cls_z_ap,
                        "dense-z AUC": dense_z_auc,
                        "dense-z AP": dense_z_ap,
                        "fixed0.7-z AUC": fixed07_auc,
                        "fixed0.7-z AP": fixed07_ap,
                        "final AUC check": final_auc,
                        "final AP check": final_ap,
                        "alpha mean": round(float(np.mean(calibrated[class_name]["alpha"])), 4),
                        "alpha std": round(float(np.std(calibrated[class_name]["alpha"])), 4),
                    })
                df.loc[len(df)] = Series(class_result_dict)
                del masks, labels, preds, preds_for_image, final_score, file_names, pack
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if not args.keep_calib_cache:
                shutil.rmtree(cache_root, ignore_errors=True)

        # Robust Average row: dynamic numeric columns, including optional V27 component metrics.
        numeric_cols = [c for c in df.columns if c != "class name"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        avg_row = {"class name": "Average"}
        avg_row.update(df[numeric_cols].mean(numeric_only=True).to_dict())
        df.loc[len(df)] = avg_row
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))





if __name__ == "__main__":
    main()
