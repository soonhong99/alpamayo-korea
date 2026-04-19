#!/usr/bin/env bash
# download_datasets.sh — Korean dataset download helper
# Usage:
#   bash scripts/download_datasets.sh --source nvidia_sample
#   bash scripts/download_datasets.sh --source aihub --dataset 188
#   bash scripts/download_datasets.sh --source kakao
#   bash scripts/download_datasets.sh --source all
set -euo pipefail

SOURCE=""
DATASET_ID=""

# ── Argument parsing ──────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --source)   SOURCE="$2";    shift 2 ;;
    --dataset)  DATASET_ID="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$SOURCE" ]]; then
  echo "Usage: bash scripts/download_datasets.sh --source <nvidia_sample|aihub|kakao|42dot|all>"
  exit 1
fi

mkdir -p data/aihub/raw data/kakao/raw data/42dot/raw data/nvidia_physicalai/sample data/nvidia_physicalai/nurec

# ── NVIDIA Physical AI (sample) ───────────────
download_nvidia_sample() {
  echo "[NVIDIA] Downloading Physical AI AV dataset sample..."
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set. Run: export HF_TOKEN='your_token'"
    exit 1
  fi

  pip install physical_ai_av --quiet 2>/dev/null || \
    pip install huggingface_hub --quiet

  python3 - <<'EOF'
import os
from huggingface_hub import snapshot_download
token = os.environ["HF_TOKEN"]
snapshot_download(
    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
    repo_type="dataset",
    local_dir="data/nvidia_physicalai/sample",
    token=token,
    ignore_patterns=["*.tar.gz"],  # skip large archives for sample
    max_workers=4,
)
print("✓ NVIDIA sample downloaded to data/nvidia_physicalai/sample/")
EOF
}

# ── NVIDIA NuRec scenes (for AlpaSim) ─────────
download_nvidia_nurec() {
  echo "[NVIDIA NuRec] Downloading NuRec scenes for AlpaSim..."
  python3 - <<'EOF'
import os
from huggingface_hub import snapshot_download
token = os.environ.get("HF_TOKEN")
snapshot_download(
    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles-NuRec",
    repo_type="dataset",
    local_dir="data/nvidia_physicalai/nurec",
    token=token,
    max_workers=4,
)
print("✓ NuRec scenes downloaded to data/nvidia_physicalai/nurec/")
EOF
}

# ── AI Hub ────────────────────────────────────
download_aihub() {
  echo "[AI Hub] Downloading dataset..."
  echo ""
  echo "  AI Hub requires manual authentication."
  echo "  Steps:"
  echo "  1. Go to https://aihub.or.kr"
  echo "  2. Search for dataset ID: ${DATASET_ID:-188}"
  echo "  3. Click '신청하기' and fill in the form"
  echo "  4. After approval (1-3 days), use the API:"
  echo ""
  echo "  export AIHUB_ID='your_email@example.com'"
  echo "  export AIHUB_PW='your_password'"
  echo ""
  echo "  Then re-run this script."
  echo ""

  if [[ -z "${AIHUB_ID:-}" ]] || [[ -z "${AIHUB_PW:-}" ]]; then
    echo "  AIHUB_ID and AIHUB_PW not set — skipping download."
    echo "  Set them and re-run to proceed."
    return
  fi

  DID="${DATASET_ID:-188}"
  OUT="data/aihub/raw/dataset_${DID}.zip"

  curl -o "$OUT" \
    -H "Authorization: Bearer $(python3 -c "
import requests, os
r = requests.post('https://api.aihub.or.kr/api/auth/login',
  json={'loginId': os.environ['AIHUB_ID'], 'password': os.environ['AIHUB_PW']})
print(r.json().get('token',''))
")" \
    "https://api.aihub.or.kr/api/down/1.0/${DID}.do?fileSn=all" || {
    echo "  Download failed — check credentials and dataset approval status."
    return
  }

  echo "  ✓ AI Hub dataset ${DID} saved to ${OUT}"
  echo "  Extracting..."
  unzip -q "$OUT" -d "data/aihub/raw/" && rm "$OUT"
  echo "  ✓ Extracted to data/aihub/raw/"
}

# ── Kakao Mobility (ETRI AI 나눔) ─────────────
download_kakao() {
  echo "[Kakao Mobility] Instructions for ETRI AI 나눔 dataset:"
  echo ""
  echo "  1. Go to https://nanum.etri.re.kr"
  echo "  2. Register (free, no copyright restrictions)"
  echo "  3. Search for '카카오모빌리티 자율주행'"
  echo "  4. Download (150K samples, ~30GB)"
  echo "  5. Place files in data/kakao/raw/"
  echo ""
  echo "  Direct download URL (requires login cookie):"
  echo "  https://nanum.etri.re.kr/share/list?category=AutoDriving"
  echo ""
  echo "  After download, run preprocessing:"
  echo "  python scripts/preprocess_kakao.py --input data/kakao/raw/ --output data/kakao/"
}

# ── 42dot ─────────────────────────────────────
download_42dot() {
  echo "[42dot] Instructions for 42dot AKit dataset:"
  echo ""
  echo "  1. Go to https://42dot.ai/akit"
  echo "  2. Fill in the access request form"
  echo "  3. After approval, download multi-camera + LiDAR data"
  echo "  4. Place files in data/42dot/raw/"
}

# ── Dispatch ──────────────────────────────────
case "$SOURCE" in
  nvidia_sample)  download_nvidia_sample ;;
  nvidia_nurec)   download_nvidia_nurec ;;
  aihub)          download_aihub ;;
  kakao)          download_kakao ;;
  42dot)          download_42dot ;;
  all)
    download_nvidia_sample
    download_aihub
    download_kakao
    download_42dot
    ;;
  *)
    echo "Unknown source: $SOURCE"
    echo "Valid sources: nvidia_sample, nvidia_nurec, aihub, kakao, 42dot, all"
    exit 1
    ;;
esac

echo ""
echo "Done."
