# Data Directory

This directory holds all datasets used for Alpamayo-Korea training and evaluation.

**All subdirectories are excluded from git** (see `.gitignore`). Never commit dataset files — they are large and may have license restrictions.

---

## Directory Structure

```
data/
├── aihub/
│   ├── raw/            ← Downloaded ZIP files from AI Hub API
│   ├── images/         ← Extracted JPEG/PNG images
│   └── annotations/    ← COCO-format JSON annotation files
│
├── kakao/
│   ├── raw/            ← Downloaded archives from ETRI AI 나눔
│   ├── lidar/          ← LiDAR point cloud files (.bin / .pcd)
│   ├── camera/         ← Camera images
│   └── labels/         ← Object detection labels
│
├── 42dot/
│   └── raw/            ← Multi-camera + LiDAR fusion data
│
└── nvidia_physicalai/
    ├── sample/         ← Small sample for quick testing (download first)
    └── nurec/          ← NuRec scene reconstruction artifacts for AlpaSim
```

---

## Download Instructions

See `docs/datasets.md` for full download guide.

Quick start:

```bash
# 1. NVIDIA sample (needed for AlpaSim — do this first)
export HF_TOKEN="your_token"
bash scripts/download_datasets.sh --source nvidia_sample

# 2. AI Hub (Korean traffic light/sign data)
export AIHUB_ID="your_email"
export AIHUB_PW="your_password"
bash scripts/download_datasets.sh --source aihub --dataset 188

# 3. Kakao Mobility (manual download from nanum.etri.re.kr)
bash scripts/download_datasets.sh --source kakao
# (Follow the printed instructions)
```

---

## Storage Estimates

| Dataset | Estimated Size |
|---|---|
| AI Hub #188 | ~50GB (1.9M images + annotations) |
| Kakao Mobility | ~30GB (150K LiDAR + camera samples) |
| 42dot | ~20GB (multi-camera sequences) |
| NVIDIA Physical AI sample | ~5GB |
| NVIDIA Physical AI NuRec | ~100GB+ |

**Recommended storage**: ≥500GB SSD for development, ≥1TB for full dataset.
