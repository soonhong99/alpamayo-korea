# 실시간 스트리밍 추론 — Thor 배포 가이드

**작성 일시**: 2026-05-19  
**대상**: Jetson AGX Thor (ice401@100.95.177.101)

---

## 1. "0.4초 동영상"이란 무엇인가

```
오해: 모델에 0.4초짜리 짧은 동영상 파일을 넣는다
실제: 슬라이딩 윈도우 — 긴 동영상에서 항상 가장 최근 0.4초(4 프레임)만 잘라서 입력

예시 (20초 클립):
  t=0.4s:  [f0, f1, f2, f3] → 추론 → 6.4초 미래 궤적
  t=0.5s:  [f1, f2, f3, f4] → 추론 → 6.4초 미래 궤적   (f0 버려짐)
  t=0.6s:  [f2, f3, f4, f5] → 추론 → ...
  ...
  t=20s:   마지막 4프레임 → 추론

→ 즉, 입력 동영상은 얼마나 길어도 됨.
  0.4초는 모델이 한 번에 보는 과거 시야의 크기일 뿐.
```

---

## 2. 사용 가능한 영상 데이터 소스

### 옵션 A: NVIDIA PhysicalAI-AV 데이터셋 ★ 권장

```
분량: 306,152 클립 × 20초 = 약 1,700시간
형식: MP4 (1080p, 30fps) + egomotion Parquet
카메라: 7대 (front-wide, front-tele, side 등)
장점: 실제 egomotion(xyz, rot) 데이터 포함 — 모델 정확도 최대
단점: 총 133 TB, HuggingFace 토큰 + 라이선스 동의 필요

접근:
  1. https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
  2. 라이선스 동의 → HF 토큰 생성
  3. pip install physical_ai_av
  4. 클립 1개(약 200-500 MB)만 선택 다운로드 가능
```

### 옵션 B: 블랙박스 / 일반 주행 영상 ★ 지금 당장 가능

```
어떤 MP4 주행 동영상이든 사용 가능.
egomotion은 Mock(직진 시뮬레이션)으로 대체 — 파이프라인 테스트에 충분.

예:
  - 본인 블랙박스 영상 (MP4)
  - YouTube 주행 영상 다운로드 (yt-dlp 사용)
  - 국내 AI Hub 데이터셋의 MP4 파일

단점: 실제 GPS/IMU 없음 → egomotion이 정확하지 않음
     (궤적 정확도보다 파이프라인 동작 확인 목적)
```

### 옵션 C: Mock (데이터 없이 즉시 테스트)

```
랜덤 텐서로 파이프라인 동작만 확인.
latency 벤치마크에 사용.
```

---

## 3. PC → Thor 전송 파일 목록

### WSL 터미널에서 실행

**필수: 새 스트리밍 스크립트**
```bash
scp /mnt/c/Users/nanay/Desktop/Alphamayo/scripts/run_streaming_inference.py \
    ice401@100.95.177.101:~/alpamayo1.5/scripts/
```

**영상 파일이 있는 경우 (옵션)**
```bash
# 단일 MP4 (블랙박스 등)
scp /mnt/c/Users/nanay/Desktop/driving_clip.mp4 \
    ice401@100.95.177.101:~/alpamayo1.5/data/test_videos/

# PhysicalAI 클립 여러 개
scp /mnt/c/Users/nanay/Desktop/physicalai/*.mp4 \
    ice401@100.95.177.101:~/alpamayo1.5/data/test_videos/
```

**문서 (참고용, 선택)**
```bash
scp /mnt/c/Users/nanay/Desktop/Alphamayo/docs/260519_alpamayo_input_format_realtime.md \
    ice401@100.95.177.101:~/alpamayo1.5/docs/
```

---

## 4. Thor에서 실행 순서 (전체)

### Step 1: SSH 접속
```bash
ssh ice401@100.95.177.101
```

### Step 2: venv 활성화 (항상 먼저)
```bash
source ~/alpamayo1.5/a1_5_venv/bin/activate

# 확인 — 프롬프트가 (a1_5_venv) 로 바뀌어야 함
which python3
# → /home/ice401/alpamayo1.5/a1_5_venv/bin/python3
```

### Step 3: 의존성 설치 (최초 1회)
```bash
# 동영상 파일 읽기용 (MP4 소스 사용할 경우 필수)
python3 -m pip install opencv-python-headless

# PhysicalAI 데이터셋 사용할 경우
python3 -m pip install physical_ai_av

# 설치 확인
python3 -c "import cv2; print(cv2.__version__)"
```

### Step 4: 테스트 디렉토리 생성
```bash
mkdir -p ~/alpamayo1.5/data/test_videos
mkdir -p ~/alpamayo1.5/evaluation/results/streaming
```

### Step 5-A: Mock 모드로 파이프라인 동작 확인 (모델 없이)
```bash
# alpamayo1_5 패키지 없어도 Mock 추론으로 동작
python3 ~/alpamayo1.5/scripts/run_streaming_inference.py \
    --source mock \
    --hz 10 \
    --bench \
    --iterations 30

# 정상 출력 예:
# [0001] ✅ latency=52.1ms | wp[0]=(0.85,0.00)m wp[5]=(3.21,0.00)m
# [0002] ✅ latency=50.8ms | ...
# ...
# 평균 latency: 51.2ms  P95: 55.1ms
```

### Step 5-B: MP4 동영상으로 실행

```bash
# 단일 MP4 → 4 카메라 모두 같은 영상 사용
python3 ~/alpamayo1.5/scripts/run_streaming_inference.py \
    --source ~/alpamayo1.5/data/test_videos/driving_clip.mp4 \
    --hz 10 \
    --dtype bf16 \
    --lang ko

# 영상이 끝나도 자동으로 처음부터 반복됨 (loop=True)
```

### Step 5-C: 실제 Alpamayo 모델로 실행 (모델 다운로드 후)

```bash
# HF 토큰 설정 (최초 1회)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 모델 다운로드 (22 GB, 최초 1회 — 시간 오래 걸림)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'nvidia/Alpamayo-1.5-10B',
    local_dir='/home/ice401/alpamayo1.5/checkpoints/alpamayo_base',
    token='$HF_TOKEN',
)
print('다운로드 완료')
"

# 모델로 스트리밍 실행
python3 ~/alpamayo1.5/scripts/run_streaming_inference.py \
    --model ~/alpamayo1.5/checkpoints/alpamayo_base \
    --source ~/alpamayo1.5/data/test_videos/driving_clip.mp4 \
    --hz 10 \
    --dtype fp4 \
    --lang ko \
    --save_traces
```

---

## 5. PhysicalAI 클립 1개 다운로드 방법 (Thor에서 직접)

20초짜리 클립 1개만 받아서 테스트하는 방법:

```bash
source ~/alpamayo1.5/a1_5_venv/bin/activate
python3 -m pip install physical_ai_av huggingface_hub

export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

python3 << 'EOF'
from physical_ai_av import PhysicalAIAVDataset
import pandas as pd

# 사용 가능한 clip ID 목록 확인
dataset = PhysicalAIAVDataset(token="$HF_TOKEN")
clip_ids = dataset.get_clip_ids()
print(f"총 클립 수: {len(clip_ids)}")

# 클립 1개 다운로드 (약 200-500 MB)
clip_id = clip_ids[0]
print(f"다운로드: {clip_id}")
dataset.download_clip(
    clip_id,
    sensors=["camera_front_wide_120fov"],  # 필요한 카메라만
    output_dir="/home/ice401/alpamayo1.5/data/test_videos/",
)
print("완료")
EOF
```

---

## 6. 트러블슈팅

### "alpamayo1_5 패키지 없음" 오류
```bash
# alpamayo1.5 소스 설치 (NVlabs GitHub 클론 후)
cd ~/alpamayo1.5
pip install -e .
# 또는
python3 -m pip install -e .
```

### "opencv 없음" 오류 (MP4 소스 사용 시)
```bash
python3 -m pip install opencv-python-headless
# headless 버전 사용 — Thor에는 GUI 없음
```

### 추론이 목표 Hz보다 느릴 때
```
경고: 추론(XXXms) > 목표 인터벌(100ms)

해결 방법:
1. --dtype fp4 로 변경 (BF16 대비 4× 빠름)
2. --hz 5 로 낮춤 (200ms 여유)
3. EXP-3 크로스프레임 파이프라인 활성화 (Vision 병렬 실행)
```

### egomotion 데이터 없을 때 (MP4 소스)
```
MP4 파일에는 GPS/IMU 없음 → Mock egomotion 자동 사용
모델이 동작은 하지만 궤적 정확도 낮을 수 있음
PhysicalAI 클립 사용 시 실제 egomotion 이용 가능
```

---

## 7. 파일 요약

| 파일 | 위치 | 용도 |
|---|---|---|
| `run_streaming_inference.py` | `scripts/` | 실시간 스트리밍 추론 메인 |
| `run_thor_inference.py` | `scripts/` | 기존 단발성 추론 (shape 오류 있음, 참고용) |
| `260519_alpamayo_input_format_realtime.md` | `docs/` | 입력 포맷 치트시트 |
| `260519_streaming_thor_deploy_guide.md` | `docs/` | 이 파일 |
