# DESR: DINO-Enhanced Score Reliability for Cross-Dataset Industrial Anomaly Transfer

This repository contains the official implementation of **DESR**, a source-trained industrial anomaly transfer framework that combines CLIP semantic evidence with DINOv3 dense structural priors for cross-dataset anomaly detection and localization.

DESR is designed for **source-only transfer**: the model is trained on one industrial anomaly dataset and evaluated directly on another target dataset without using target-domain labels for training.

## Highlights

- **DINO-enhanced dense evidence**: a frozen DINOv3 dense-token branch injects structural priors into CLIP patch representations.
- **CLIP text/image adaptation**: lightweight text and image adapters are trained while the foundation encoders remain frozen.
- **Dense-to-global scoring**: image-level anomaly scores combine dense top-k evidence and CLIP global evidence.
- **Cross-dataset evaluation**: supports transfer among MVTec-AD, VisA, BTAD, MPDD, DTD, and DAGM through metadata files.
- **Qualitative visualization**: supports heatmap generation with GT contours for paper-ready inspection figures.

## Repository structure

```text
DESR/
├── dataset/
│   ├── constants.py              # Dataset roots, class names, prompts
│   └── metadata/                 # JSONL metadata for train/test splits
├── model/
│   ├── adapter.py                # DESR adapter model with optional DINOv3 fusion
│   ├── adapter_modules.py        # Adapter and projection modules
│   ├── clip.py                   # CLIP model loading utilities
│   ├── ViT-L-14-336px.pt         # Place the CLIP ViT-L/14@336px checkpoint here
│   └── ...
├── forward_utils.py              # Similarity maps, metrics, visualization
├── train.py                      # Stage-1 text and Stage-2 image training
├── test.py                       # Cross-dataset evaluation and visualization
└── utils.py
```

## Environment

The code was developed with PyTorch and CUDA GPUs. A typical environment is:

```bash
conda create -n desr python=3.10 -y
conda activate desr

# Install PyTorch following your CUDA version from the official PyTorch website.
# Example only:
pip install torch torchvision torchaudio

pip install numpy pandas scikit-learn tqdm pillow opencv-python kornia ftfy regex ipdb
pip install "transformers>=4.56.0" huggingface_hub
```

DINOv3 is loaded through Hugging Face Transformers when `--use_dinov3` is enabled. If your environment requires Hugging Face authentication, log in before running:

```bash
huggingface-cli login
```

## Model checkpoint preparation

Before training or testing, place the CLIP ViT-L/14@336px checkpoint in the `model/` directory:

```text
DESR/
└── model/
    └── ViT-L-14-336px.pt
```

The expected path is:

```text
model/ViT-L-14-336px.pt
```

This path is used by `model/clip.py` for `--model_name ViT-L-14-336`. If the file is missing, CLIP initialization will fail or the pretrained weights will not be loaded correctly.

The DINOv3 branch is loaded separately through Hugging Face Transformers when `--use_dinov3` is enabled.

## Dataset preparation

DESR uses JSONL metadata files under `dataset/metadata/<DATASET>/`. The actual image roots are configured in `dataset/constants.py` through the `DATA_PATH` dictionary.

You can either edit `dataset/constants.py`:

```python
DATA_PATH = {
    "MVTec": "path/to/mvtec_anomaly_detection",
    "VisA": "path/to/VisA",
    "BTAD": "path/to/BTAD",
    "MPDD": "path/to/MPDD",
    "DTD": "path/to/DTD-Synthetic",
    "DAGM": "path/to/DAGM",
}
```

or override the path at runtime:

```bash
python test.py --dataset MVTec --data_path path/to/mvtec_anomaly_detection ...
```

Expected MVTec layout:

```text
mvtec_anomaly_detection/
├── bottle/
├── cable/
├── capsule/
└── ...
```


## Released checkpoints

Large checkpoint files are not included in this GitHub repository. The released DESR checkpoint files can be downloaded from Google Drive:

```text
https://drive.google.com/drive/folders/1ityUyN2aPki17Kz30epoOR6YMJumJAOq?usp=sharing
```

For reproducing the reported VisA → MVTec result, download the checkpoint files and place them under:

```text
ckpt/visa_desr/
├── text_adapter.pth
├── image_adapter_best_source.pth
└── image_adapter_6.pth
```

The main evaluation command uses `--eval_best_source`, which loads `image_adapter_best_source.pth`. The explicit epoch-6 command uses `image_adapter_6.pth`.

If you evaluate MVTec → VisA, place the corresponding MVTec-source checkpoint files under:

```text
ckpt/mvtec_desr/
├── text_adapter.pth
├── image_adapter_best_source.pth
└── image_adapter_6.pth
```

Recommended GitHub practice: keep `ckpt/` and `*.pth` ignored in `.gitignore`, and provide checkpoint files through Google Drive, Hugging Face, or GitHub Releases.

## Checkpoint directory convention

The original experiment directory name was long. For GitHub usage, this README uses short checkpoint folders:

```text
ckpt/visa_desr      # VisA source checkpoint, used for VisA -> MVTec/BTAD/DTD/DAGM/MPDD
ckpt/mvtec_desr     # MVTec source checkpoint, used for MVTec -> VisA
ckpt/visa_clip      # optional CLIP-only / no-DINO checkpoint
```


All commands below use the shorter `ckpt/visa_desr` and `ckpt/mvtec_desr` paths.

## Training

DESR uses two training stages:

1. **Text-side adaptation**: trains the text adapter using source-domain data.
2. **Image-side adaptation**: trains image adapters, multi-scale fusion, and optional DINOv3 projection while CLIP/DINO foundation encoders remain frozen.

### Train VisA source checkpoint

This trains the short VisA source checkpoint folder `ckpt/visa_desr`, which is used for VisA → MVTec transfer.

```bash
python train.py \
  --dataset VisA \
  --training_mode full_shot \
  --batch_size 2 \
  --text_batch_size 16 \
  --save_path ckpt/visa_desr \
  --model_name ViT-L-14-336 \
  --img_size 518 \
  --text_epoch 5 \
  --image_epoch 6 \
  --text_lr 0.000005 \
  --image_lr 0.00003 \
  --text_norm_weight 0.05 \
  --text_anchor_sep_weight 0.02 \
  --text_anchor_sep_max_cos 0.15 \
  --text_anchor_preserve_weight 0.08 \
  --text_adapt_weight 0.08 \
  --image_adapt_weight 0.04 \
  --text_adapt_until 3 \
  --image_adapt_until 6 \
  --amp \
  --grad_accum 4 \
  --use_adapter_ema \
  --adapter_ema_decay 0.999 \
  --save_ema_as_eval \
  --use_dinov3 \
  --dino_model_name facebook/dinov3-vith16plus-pretrain-lvd1689m \
  --dino_fusion_alpha 0.05 \
  --use_score_calibrator \
  --score_calib_alpha_min 0.85 \
  --score_calib_alpha_max 1.0 \
  --image_calib_loss_weight 0.02 \
  --topk_ratio 0.01 \
  --save_best_source_checkpoint
```

Important outputs:

```text
ckpt/visa_desr/text_adapter.pth
ckpt/visa_desr/image_adapter_6.pth
ckpt/visa_desr/image_adapter_best_source.pth
ckpt/visa_desr/train.log
```

### Optional: train a CLIP-only / no-DINO checkpoint

For the no-DINO test command, train a separate checkpoint without `--use_dinov3`:

```bash
python train.py \
  --dataset VisA \
  --training_mode full_shot \
  --batch_size 2 \
  --text_batch_size 16 \
  --save_path ckpt/visa_clip \
  --model_name ViT-L-14-336 \
  --img_size 518 \
  --text_epoch 5 \
  --image_epoch 6 \
  --text_lr 0.000005 \
  --image_lr 0.00003 \
  --text_norm_weight 0.05 \
  --text_anchor_sep_weight 0.02 \
  --text_anchor_sep_max_cos 0.15 \
  --text_anchor_preserve_weight 0.08 \
  --text_adapt_weight 0.08 \
  --image_adapt_weight 0.04 \
  --text_adapt_until 3 \
  --image_adapt_until 6 \
  --amp \
  --grad_accum 4 \
  --use_adapter_ema \
  --adapter_ema_decay 0.999 \
  --save_ema_as_eval \
  --topk_ratio 0.01 \
  --save_best_source_checkpoint
```

## Evaluation

### VisA → MVTec final scoring setting

This command evaluates the VisA source checkpoint on MVTec using the final scoring setting reported in the paper.

```bash
python test.py \
  --dataset MVTec \
  --shot 4 \
  --batch_size 2 \
  --save_path ckpt/visa_desr \
  --model_name ViT-L-14-336 \
  --img_size 518 \
  --eval_best_source \
  --use_dinov3 \
  --dino_model_name facebook/dinov3-vith16plus-pretrain-lvd1689m \
  --dino_fusion_alpha 0.05 \
  --fb_bg_suppression_beta 0.2 \
  --fb_bg_suppression_mode normal_z \
  --image_fusion_alpha 0.80 \
  --topk_ratio 0.006
```

The expected MVTec average row for the released/source-best checkpoint is approximately:

| Target | Pixel-AUC | PRO | Image-AUC | Image-AP | S4 |
|---|---:|---:|---:|---:|---:|
| MVTec | 92.3 | 89.2 | 93.6 | 97.2 | 93.1 |

where:

```text
S4 = (Pixel-AUC + PRO + Image-AUC + Image-AP) / 4
```

### Explicit epoch-6 evaluation

To evaluate epoch 6 directly instead of `image_adapter_best_source.pth`:

```bash
python test.py \
  --dataset MVTec \
  --shot 4 \
  --batch_size 2 \
  --save_path ckpt/visa_desr \
  --model_name ViT-L-14-336 \
  --img_size 518 \
  --eval_epochs 6 \
  --use_dinov3 \
  --dino_model_name facebook/dinov3-vith16plus-pretrain-lvd1689m \
  --dino_fusion_alpha 0.05 \
  --fb_bg_suppression_beta 0.2 \
  --fb_bg_suppression_mode normal_z \
  --image_fusion_alpha 0.80 \
  --topk_ratio 0.006
```

### MVTec → VisA transfer

For the reverse direction using the MVTec source checkpoint:

```bash
python test.py \
  --dataset VisA \
  --shot 4 \
  --batch_size 2 \
  --save_path ckpt/mvtec_desr \
  --model_name ViT-L-14-336 \
  --img_size 518 \
  --eval_best_source \
  --use_dinov3 \
  --dino_model_name facebook/dinov3-vith16plus-pretrain-lvd1689m \
  --dino_fusion_alpha 0.05 \
  --fb_bg_suppression_beta 0.2 \
  --fb_bg_suppression_mode normal_z \
  --image_fusion_alpha 0.90 \
  --topk_ratio 0.01
```

## Cross-dataset evaluation

Example commands for evaluating a VisA source checkpoint on additional target datasets:

```bash
# BTAD
python test.py --dataset BTAD --shot 4 --batch_size 2 --save_path ckpt/visa_desr --model_name ViT-L-14-336 --img_size 518 --eval_best_source --use_dinov3 --dino_model_name facebook/dinov3-vith16plus-pretrain-lvd1689m --dino_fusion_alpha 0.05 --fb_bg_suppression_beta 0.2 --fb_bg_suppression_mode normal_z --image_fusion_alpha 0.80 --topk_ratio 0.006

# DTD
python test.py --dataset DTD --shot 4 --batch_size 2 --save_path ckpt/visa_desr --model_name ViT-L-14-336 --img_size 518 --eval_best_source --use_dinov3 --dino_model_name facebook/dinov3-vith16plus-pretrain-lvd1689m --dino_fusion_alpha 0.05 --fb_bg_suppression_beta 0.2 --fb_bg_suppression_mode normal_z --image_fusion_alpha 0.80 --topk_ratio 0.006

# DAGM
python test.py --dataset DAGM --shot 4 --batch_size 2 --save_path ckpt/visa_desr --model_name ViT-L-14-336 --img_size 518 --eval_best_source --use_dinov3 --dino_model_name facebook/dinov3-vith16plus-pretrain-lvd1689m --dino_fusion_alpha 0.05 --fb_bg_suppression_beta 0.2 --fb_bg_suppression_mode normal_z --image_fusion_alpha 0.80 --topk_ratio 0.006
```

## Visualization

Add `--visualize` to save heatmap panels.

```bash
python test.py \
  --dataset MVTec \
  --shot 4 \
  --batch_size 2 \
  --save_path ckpt/visa_desr \
  --model_name ViT-L-14-336 \
  --img_size 518 \
  --eval_best_source \
  --use_dinov3 \
  --dino_model_name facebook/dinov3-vith16plus-pretrain-lvd1689m \
  --dino_fusion_alpha 0.05 \
  --fb_bg_suppression_beta 0.2 \
  --fb_bg_suppression_mode normal_z \
  --image_fusion_alpha 0.80 \
  --topk_ratio 0.006 \
  --visualize
```

Output path:

```text
ckpt/visa_desr/visualization/MVTec/<class_name>/*.png
```

## Useful ablations

### DINO residual on/off

```bash
# DINO off
python test.py --dataset MVTec --shot 4 --batch_size 2 --save_path ckpt/visa_desr --model_name ViT-L-14-336 --img_size 518 --eval_best_source --use_dinov3 --dino_model_name facebook/dinov3-vith16plus-pretrain-lvd1689m --dino_fusion_alpha 0.0 --fb_bg_suppression_beta 0.2 --fb_bg_suppression_mode normal_z --image_fusion_alpha 0.80 --topk_ratio 0.006

# DINO on
python test.py --dataset MVTec --shot 4 --batch_size 2 --save_path ckpt/visa_desr --model_name ViT-L-14-336 --img_size 518 --eval_best_source --use_dinov3 --dino_model_name facebook/dinov3-vith16plus-pretrain-lvd1689m --dino_fusion_alpha 0.05 --fb_bg_suppression_beta 0.2 --fb_bg_suppression_mode normal_z --image_fusion_alpha 0.80 --topk_ratio 0.006
```

### No-DINO / CLIP-only test command

If you want to evaluate a checkpoint trained **without** the DINOv3 branch, use a separate CLIP-only checkpoint directory, for example `ckpt/visa_clip`. Do not use the DINO-trained `ckpt/visa_desr` checkpoint without `--use_dinov3`, because the released DINO checkpoint contains DINO projection weights and `test.py` will require matching DINO settings.

```bash
python test.py \
  --dataset MVTec \
  --shot 4 \
  --batch_size 2 \
  --save_path ckpt/visa_clip \
  --model_name ViT-L-14-336 \
  --img_size 518 \
  --eval_best_source \
  --fb_bg_suppression_beta 0.2 \
  --fb_bg_suppression_mode normal_z \
  --image_fusion_alpha 0.80 \
  --topk_ratio 0.006
```

For a DINO-trained checkpoint where you only want to disable the residual contribution, keep `--use_dinov3` but set `--dino_fusion_alpha 0.0` as shown in the DINO residual off ablation above.

### Epoch sweep

```bash
python test.py \
  --dataset MVTec \
  --shot 4 \
  --batch_size 2 \
  --save_path ckpt/visa_desr \
  --model_name ViT-L-14-336 \
  --img_size 518 \
  --eval_epochs 1,2,3,4,5,6 \
  --use_dinov3 \
  --dino_model_name facebook/dinov3-vith16plus-pretrain-lvd1689m \
  --dino_fusion_alpha 0.05 \
  --fb_bg_suppression_beta 0.2 \
  --fb_bg_suppression_mode normal_z \
  --image_fusion_alpha 0.80 \
  --topk_ratio 0.006
```

## Notes

- Datasets and large checkpoints are not included in this repository.
- The final MVTec score depends on the checkpoint state and final scoring parameters. Retraining may introduce small variations.
- `--eval_best_source` evaluates `image_adapter_best_source.pth`, which is selected using a source-only proxy during training.
- If a checkpoint was trained with DINOv3 projection weights, test with `--use_dinov3` and the same DINO settings.
- For pure no-DINO evaluation, use a checkpoint trained without `--use_dinov3`, such as `ckpt/visa_clip`.

## Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{lin2026desr,
  title     = {DESR: DINO-Enhanced Score Reliability for Cross-Dataset Industrial Anomaly Transfer},
  author    = {Lin, Shih-Chih and Hwang, Jenq-Neng and Lai, Shang-Hong},
  booktitle = {Proceedings of the ECML-PKDD},
  year      = {2026}
}
```

## Acknowledgements

This codebase builds on CLIP-style vision-language anomaly detection pipelines and uses frozen CLIP and DINOv3 foundation encoders with lightweight adapters for industrial anomaly transfer.
