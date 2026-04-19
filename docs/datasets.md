# Korean Autonomous Driving Datasets

Complete guide to obtaining and using Korean road datasets for Alpamayo-Korea.

---

## 1. AI Hub (aihub.or.kr) — Primary Source

**관리기관:** 한국지능정보사회진흥원 (NIA)
**URL:** https://aihub.or.kr
**접근:** 회원가입 → 데이터 신청 → 1–3일 승인

### Key Datasets

| Dataset ID | Name | Size | Contents |
|---|---|---|---|
| #188 | 도로주행 영상 (신호등·표지판) | 1.9M images | Traffic lights (horizontal!), signs, bboxes |
| #도로주행 | 자율주행 도로주행 | Multi-sensor | Camera + LiDAR, urban/highway |
| #보행자 인식 | 보행자 인식 데이터 | 500K+ images | Pedestrians, crosswalk behavior |
| #야간 주행 | 야간·악천후 주행 | 200K+ images | Night, rain, fog conditions |

### Download Steps

1. Go to https://aihub.or.kr and create account (휴대폰 본인인증 required)
2. Search for dataset, click "신청하기"
3. Fill in usage purpose form (학술 연구 / 포트폴리오 프로젝트)
4. After approval email, use API download:

```bash
export AIHUB_ID="your_email"
export AIHUB_PW="your_password"

# Example: Download dataset #188 (traffic light/sign data)
bash scripts/download_datasets.sh --source aihub --dataset 188
```

### Format
- Images: JPEG/PNG
- Annotations: JSON (COCO-style bounding boxes)
- Coordinate system: pixel coordinates with class labels

### Why This Dataset Is Critical
AI Hub dataset #188 contains **Korean horizontal traffic lights** — the exact scenario where Alpamayo baseline fails. The annotation includes signal state (red/green/yellow/arrow) which can be used to create reasoning supervision.

---

## 2. Kakao Mobility / ETRI AI 나눔

**URL:** https://nanum.etri.re.kr
**접근:** 회원가입 후 즉시 다운로드 가능 (저작권 없음)
**공개일:** 2025년 2월 28일

### Contents
- 150,000 samples
- 10 object categories: person, vehicle, bicycle + traffic light, sign
- 3D dynamic objects (LiDAR point clouds)
- 2D static objects (camera images)
- Collected from: 국내 주요 도로변 설치 LiDAR·카메라 엣지 인프라

### Why This Is Useful
Kakao Mobility's data was collected specifically for **urban edge infrastructure** — the same deployment context as Jetson Thor. This makes it ideal for evaluating real-world Korean deployment readiness.

### Download

```bash
bash scripts/download_datasets.sh --source kakao
# Downloads to data/kakao/
```

---

## 3. 42dot Open Dataset

**회사:** 42dot (현대차그룹 계열 자율주행 스타트업)
**URL:** https://42dot.ai/akit
**접근:** 신청 후 다운로드

### Contents
- Multi-camera + LiDAR fusion data
- Korean urban environments (Seoul metropolitan area)
- Multi-object tracking with cross-camera IDs

### Why This Is Useful
42dot data includes **multi-camera tracking** which aligns with Alpamayo 1.5's 7-camera input format.

---

## 4. NVIDIA Physical AI AV Dataset

**URL:** https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
**NuRec Scenes:** https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
**접근:** HuggingFace 계정 + 라이선스 동의 (비상업 연구용)

### Contents
- 1,727 hours of driving data
- 25 countries, 2,500+ cities
- 7 cameras + LiDAR + up to 10 radars
- **NuRec format**: scene reconstruction artifacts for AlpaSim

### Korean Coverage
Korean scenes are **sparsely represented**. This dataset is used as:
1. **Baseline comparison**: measure Alpamayo performance on global data vs Korean data
2. **NuRec scenes**: load into AlpaSim for closed-loop evaluation
3. **Transfer learning foundation**: fine-tune FROM this data TO Korean data

### Download

```bash
export HF_TOKEN="your_token"
huggingface-cli login

# Physical AI AV dataset (developer kit)
pip install physical_ai_av
python -c "from physical_ai_av import download; download(subset='sample')"

# NuRec scenes for AlpaSim
huggingface-cli download nvidia/PhysicalAI-Autonomous-Vehicles-NuRec \
  --local-dir data/nvidia_physicalai/nurec/
```

---

## 5. ETRI 공공데이터포털

**URL:** https://www.data.go.kr
**기관:** 한국전자통신연구원 (ETRI)

### Datasets
- 자율주행 차량 주행궤적 데이터 (GPS/IMU trajectories, Genesis G80)
- 3D 객체 검출 데이터셋 (LiDAR-based)
- 시맨틱 세그멘테이션 데이터셋

### Access
All data.go.kr datasets are freely downloadable after login.

---

## Dataset Integration Strategy

```
Phase 1 (baseline eval):
  NVIDIA Physical AI AV NuRec scenes → AlpaSim
  Measure baseline Alpamayo 1.5 on Korean scenario configs

Phase 2 (fine-tuning data prep):
  AI Hub #188       → traffic light perception supervision
  Kakao Mobility    → urban scenario diversity
  42dot             → multi-camera fusion alignment

Phase 3 (evaluation):
  All above → Korean scenario benchmark
  Compare: baseline vs fine-tuned on held-out Korean scenes
```

---

## Data Directory Structure

```
data/
├── README.md           ← data directory guide
├── .gitignore          ← All data directories excluded from git
│
├── aihub/
│   ├── raw/            ← Downloaded ZIP files
│   ├── images/         ← Extracted images
│   └── annotations/    ← COCO-format JSON files
│
├── kakao/
│   ├── raw/
│   ├── lidar/          ← Point cloud files
│   ├── camera/         ← Camera images
│   └── labels/         ← Object detection labels
│
├── 42dot/
│   └── raw/
│
└── nvidia_physicalai/
    ├── sample/         ← Small sample for quick testing
    └── nurec/          ← NuRec scene reconstruction artifacts
```

Note: All data directories are `.gitignore`d. Never commit dataset files.
