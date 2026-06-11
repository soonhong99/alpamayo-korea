"""
run_streaming_inference.py
Alpamayo 1.5 실시간 연속 추론 — 슬라이딩 윈도우 입력 파이프라인

현재 run_thor_inference.py의 문제점(잘못된 shape, 잘못된 API 호출)을 수정하고
실제 모델 API에 맞는 슬라이딩 윈도우 기반 실시간 추론을 구현한다.

입력 포맷 (공식 Alpamayo 1.5 API):
  image_frames:    (4 cameras, 4 frames, 3, 320, 576)   ← 0.4초 시각 이력
  ego_history_xyz: (1, 1, 16, 3)                        ← 1.6초 위치 이력
  ego_history_rot: (1, 1, 16, 3, 3)                     ← 1.6초 회전 이력
  relative_timestamps: (4, 4)                           ← t0 기준 상대 시간(초)

출력 포맷:
  pred_xyz: (1, 1, 64, 3)   ← 6.4초 미래 궤적 (64 waypoints × 10 Hz)
  extra["cot"]: str         ← Chain-of-Causation 추론 텍스트

사용법:
  # Mock 카메라 (Thor에서 즉시 테스트 가능)
  python scripts/run_streaming_inference.py --source mock --hz 10

  # 실제 USB 카메라 (추후)
  python scripts/run_streaming_inference.py --source /dev/video0 --hz 10 --lang ko

  # 벤치마크 모드 (latency 측정)
  python scripts/run_streaming_inference.py --source mock --bench --iterations 100
"""

import argparse
import json
import logging
import time
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── 상수: 공식 Alpamayo 1.5 입력 스펙 ─────────────────────────────────────
N_CAMERAS         = 4        # 카메라 수
N_FRAMES          = 4        # 카메라당 프레임 수 (0.4초 @ 10 Hz)
N_EGO_HISTORY     = 16       # egomotion 이력 스텝 수 (1.6초 @ 10 Hz)
N_FUTURE_STEPS    = 64       # 미래 예측 웨이포인트 수 (6.4초 @ 10 Hz)
FRAME_HZ          = 10       # 카메라 프레임 레이트
IMG_H, IMG_W      = 320, 576 # 전처리 후 이미지 해상도
IMG_RAW_H         = 1080     # 원본 해상도
IMG_RAW_W         = 1920

CAMERA_NAMES = ["front-wide", "front-tele", "cross-left", "cross-right"]


# ─── 슬라이딩 윈도우 버퍼 ───────────────────────────────────────────────────

class SlidingWindowBuffer:
    """
    실시간 카메라 + egomotion 데이터를 관리하는 슬라이딩 윈도우 버퍼.

    카메라 버퍼: 4 cameras × 최근 N_FRAMES 프레임 유지
    Egomotion 버퍼: 최근 N_EGO_HISTORY pose 유지

    Thread-safe: push_frame()은 캡처 스레드에서 호출하고
                 get_model_input()은 추론 스레드에서 호출해도 안전.
    """

    def __init__(
        self,
        n_cameras: int = N_CAMERAS,
        n_frames: int = N_FRAMES,
        n_ego: int = N_EGO_HISTORY,
        img_h: int = IMG_H,
        img_w: int = IMG_W,
    ):
        self.n_cameras = n_cameras
        self.n_frames = n_frames
        self.n_ego = n_ego
        self.img_h = img_h
        self.img_w = img_w

        # 각 카메라별 프레임 큐 (maxlen=n_frames → 자동 오래된 거 삭제)
        # 원소: Tensor (3, H, W), float32, 0~1 정규화
        self._cam_bufs: list[deque] = [
            deque(maxlen=n_frames) for _ in range(n_cameras)
        ]

        # Egomotion 큐
        # 원소: {"xyz": (3,), "rot": (3, 3), "ts_us": int}
        self._ego_buf: deque = deque(maxlen=n_ego)

        # 타임스탬프 큐: 카메라별 (각 프레임의 절대 ts in microseconds)
        self._ts_bufs: list[deque] = [
            deque(maxlen=n_frames) for _ in range(n_cameras)
        ]

        self._lock = threading.Lock()
        self._frame_count = 0

    @property
    def is_ready(self) -> bool:
        """최소 n_frames 프레임과 n_ego egomotion 데이터가 쌓였는지 확인."""
        with self._lock:
            cam_ready = all(len(q) == self.n_frames for q in self._cam_bufs)
            ego_ready = len(self._ego_buf) == self.n_ego
            return cam_ready and ego_ready

    def push_frame(
        self,
        frames: torch.Tensor,          # (N_cameras, 3, H, W) — 전처리된 프레임
        ego_xyz: torch.Tensor,         # (3,) — 현재 자차 위치
        ego_rot: torch.Tensor,         # (3, 3) — 현재 자차 회전
        ts_us: Optional[int] = None,   # 현재 프레임 절대 타임스탬프 (μs)
    ):
        """
        새 프레임을 버퍼에 추가. 캡처 스레드에서 호출.

        Args:
            frames: (N_cameras, 3, H, W) 전처리된 카메라 이미지 텐서
            ego_xyz: (3,) 자차 위치 [x, y, z] (ego 좌표계)
            ego_rot: (3, 3) 자차 회전 행렬
            ts_us: 타임스탬프 (마이크로초). None이면 현재 시각 사용.
        """
        if ts_us is None:
            ts_us = int(time.time() * 1e6)

        with self._lock:
            for cam_idx in range(self.n_cameras):
                self._cam_bufs[cam_idx].append(frames[cam_idx].cpu())
                self._ts_bufs[cam_idx].append(ts_us)

            self._ego_buf.append({
                "xyz": ego_xyz.cpu(),
                "rot": ego_rot.cpu(),
                "ts_us": ts_us,
            })
            self._frame_count += 1

    def get_model_input(self) -> Optional[dict]:
        """
        슬라이딩 윈도우에서 모델 입력을 구성해 반환.
        데이터가 충분하지 않으면 None 반환.

        Returns:
            dict with keys:
              image_frames:       Tensor (N_cameras, N_frames, 3, H, W)
              camera_indices:     Tensor (N_cameras,) int64
              ego_history_xyz:    Tensor (1, 1, N_ego, 3)
              ego_history_rot:    Tensor (1, 1, N_ego, 3, 3)
              relative_timestamps: Tensor (N_cameras, N_frames) float32
              absolute_timestamps: Tensor (N_cameras, N_frames) int64
              t0_us:              int (현재 프레임 절대 타임스탬프)
        """
        if not self.is_ready:
            return None

        with self._lock:
            # 카메라 이미지 스택: (N_cameras, N_frames, 3, H, W)
            image_frames = torch.stack([
                torch.stack(list(self._cam_bufs[cam_idx]), dim=0)  # (N_frames, 3, H, W)
                for cam_idx in range(self.n_cameras)
            ], dim=0)  # (N_cameras, N_frames, 3, H, W)

            # 절대 타임스탬프: (N_cameras, N_frames) int64
            abs_ts = torch.tensor([
                list(self._ts_bufs[cam_idx])
                for cam_idx in range(self.n_cameras)
            ], dtype=torch.int64)

            # t0 = 가장 최근 타임스탬프
            t0_us = int(abs_ts[0, -1].item())

            # 상대 타임스탬프: (N_cameras, N_frames) float32, 초 단위
            relative_ts = (abs_ts - t0_us).float() / 1e6

            # Egomotion: (N_ego,) 각각 xyz, rot
            ego_list = list(self._ego_buf)  # N_ego 개
            ego_xyz = torch.stack([e["xyz"] for e in ego_list], dim=0)  # (N_ego, 3)
            ego_rot = torch.stack([e["rot"] for e in ego_list], dim=0)  # (N_ego, 3, 3)

            # 공식 API shape: (1, 1, N_ego, 3), (1, 1, N_ego, 3, 3)
            ego_xyz = ego_xyz.unsqueeze(0).unsqueeze(0)  # (1, 1, N_ego, 3)
            ego_rot = ego_rot.unsqueeze(0).unsqueeze(0)  # (1, 1, N_ego, 3, 3)

            camera_indices = torch.arange(self.n_cameras, dtype=torch.int64)

        return {
            "image_frames":       image_frames,       # (4, 4, 3, 320, 576)
            "camera_indices":     camera_indices,     # (4,)
            "ego_history_xyz":    ego_xyz,            # (1, 1, 16, 3)
            "ego_history_rot":    ego_rot,            # (1, 1, 16, 3, 3)
            "relative_timestamps": relative_ts,       # (4, 4) float32
            "absolute_timestamps": abs_ts,            # (4, 4) int64
            "t0_us":              t0_us,
        }


# ─── 카메라 소스 (Mock / 실제 카메라) ────────────────────────────────────────

class MockCameraSource:
    """
    테스트용 Mock 카메라.
    실제 카메라 없이 Thor에서 바로 파이프라인을 테스트할 수 있다.

    실제 카메라로 교체 시: get_frames()만 재구현하면 됨.
    """

    def __init__(self, img_h: int = IMG_H, img_w: int = IMG_W, n_cameras: int = N_CAMERAS):
        self.img_h = img_h
        self.img_w = img_w
        self.n_cameras = n_cameras
        self._frame_idx = 0
        logger.info(f"MockCameraSource: {n_cameras}카메라 × ({img_h}×{img_w}) RGB")

    def get_frames(self) -> torch.Tensor:
        """
        현재 시각의 카메라 프레임 반환.

        Returns:
            Tensor (N_cameras, 3, H, W) float32, range [0, 1]
        """
        # 실제 카메라 구현 시 여기를 교체:
        # frames = [cap[i].read() for i in range(N_CAMERAS)]
        # frames = [preprocess(f) for f in frames]
        # return torch.stack(frames, dim=0)

        # Mock: 약간 변화하는 패턴 (완전 랜덤이 아닌 순서 있는 mock)
        self._frame_idx += 1
        frames = torch.rand(self.n_cameras, 3, self.img_h, self.img_w)
        return frames  # (4, 3, 320, 576)

    def get_egomotion(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        현재 자차 포즈 반환.

        Returns:
            ego_xyz: (3,) float32 — [x, y, z] (m)
            ego_rot: (3, 3) float32 — 회전 행렬
        """
        # Mock: 직선 주행 시뮬레이션
        t = self._frame_idx * 0.1  # 초
        ego_xyz = torch.tensor([t * 10.0, 0.0, 0.0], dtype=torch.float32)  # 10 m/s 직진
        ego_rot = torch.eye(3, dtype=torch.float32)                         # 정방향
        return ego_xyz, ego_rot


class RawResizeCameraSource(MockCameraSource):
    """
    실제 고해상도 카메라 소스 (1920×1080 → 576×320 리사이즈).
    OpenCV 설치 필요: pip install opencv-python-headless
    """

    def __init__(self, device_ids: list[int], img_h=IMG_H, img_w=IMG_W):
        super().__init__(img_h=img_h, img_w=img_w, n_cameras=len(device_ids))
        try:
            import cv2
            self._caps = [cv2.VideoCapture(dev_id) for dev_id in device_ids]
            for cap in self._caps:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_RAW_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_RAW_H)
            logger.info(f"RealCamera: {device_ids} 열기 완료")
        except ImportError:
            logger.warning("opencv-python 미설치 → Mock 모드로 폴백")
            self._caps = None

    def get_frames(self) -> torch.Tensor:
        if self._caps is None:
            return super().get_frames()  # 폴백

        import cv2
        frames = []
        for cap in self._caps:
            ret, frame = cap.read()
            if not ret:
                frames.append(torch.zeros(3, self.img_h, self.img_w))
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(frame_rgb).float() / 255.0  # (H, W, 3)
            t = t.permute(2, 0, 1)                           # (3, H, W)
            t = F.interpolate(
                t.unsqueeze(0),
                size=(self.img_h, self.img_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            frames.append(t)
        return torch.stack(frames, dim=0)  # (N_cameras, 3, H, W)


class VideoFileCameraSource(MockCameraSource):
    """
    MP4/AVI 동영상 파일을 10 Hz로 읽어 카메라 입력처럼 제공.

    PhysicalAI 데이터셋 클립(20초, 30fps MP4) 또는
    블랙박스 영상 등 어떤 주행 동영상이든 사용 가능.

    동작 방식:
      - 영상 원래 FPS 자동 감지 (e.g., 30fps)
      - 10 Hz에 맞게 매 N번째 프레임만 추출 (e.g., 30fps → 3번째마다)
      - 영상이 끝나면 처음부터 반복 (loop=True) 또는 종료
      - 카메라가 1개뿐일 경우 같은 프레임을 4개 카메라에 복제

    설치: pip install opencv-python-headless

    사용법:
      # 동영상 1개 → 4 카메라 전부에 동일 프레임 (테스트용)
      python run_streaming_inference.py --source /path/to/clip.mp4

      # 카메라별 동영상 4개 (카메라 순서: front-wide, front-tele, cross-left, cross-right)
      python run_streaming_inference.py --source "fw.mp4,ft.mp4,cl.mp4,cr.mp4"
    """

    def __init__(
        self,
        video_paths: list[str],
        target_hz: float = FRAME_HZ,
        img_h: int = IMG_H,
        img_w: int = IMG_W,
        loop: bool = True,
    ):
        n_cams = N_CAMERAS
        super().__init__(img_h=img_h, img_w=img_w, n_cameras=n_cams)
        self.loop = loop
        self._finished = False

        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            raise RuntimeError(
                "opencv-python-headless 미설치.\n"
                "  python3 -m pip install opencv-python-headless"
            )

        # 동영상 1개 → 4 카메라에 동일 적용
        if len(video_paths) == 1:
            video_paths = video_paths * N_CAMERAS

        self._caps = []
        self._frame_steps = []

        for path in video_paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise FileNotFoundError(f"동영상 열기 실패: {path}")
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_s = total_frames / src_fps if src_fps > 0 else 0
            step = max(1, round(src_fps / target_hz))  # e.g., 30fps → step=3

            self._caps.append(cap)
            self._frame_steps.append(step)
            logger.info(
                f"  동영상: {Path(path).name}  "
                f"FPS={src_fps:.1f}  frames={total_frames}  "
                f"duration={duration_s:.1f}s  "
                f"→ 10Hz step={step}"
            )

        logger.info(f"VideoFileCameraSource: {len(self._caps)}개 동영상 준비 완료")

    def get_frames(self) -> torch.Tensor:
        """10 Hz에 맞는 프레임을 추출해 반환."""
        if self._finished:
            # 루프 없이 끝난 경우 마지막 프레임 반환
            return torch.zeros(N_CAMERAS, 3, self.img_h, self.img_w)

        frames = []
        for cam_idx, (cap, step) in enumerate(zip(self._caps, self._frame_steps)):
            # step-1 프레임 스킵 (10Hz 맞추기)
            for _ in range(step - 1):
                cap.grab()  # read() 없이 빠르게 skip

            ret, frame = cap.read()
            if not ret:
                if self.loop:
                    cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        frames.append(torch.zeros(3, self.img_h, self.img_w))
                        continue
                    logger.debug(f"카메라 {cam_idx} 동영상 루프 재시작")
                else:
                    self._finished = True
                    frames.append(torch.zeros(3, self.img_h, self.img_w))
                    continue

            # BGR → RGB → Tensor → resize
            frame_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(frame_rgb.copy()).float() / 255.0  # (H, W, 3)
            t = t.permute(2, 0, 1)                                  # (3, H, W)
            t = F.interpolate(
                t.unsqueeze(0),
                size=(self.img_h, self.img_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            frames.append(t)

        return torch.stack(frames, dim=0)  # (N_cameras, 3, H, W)

    def get_egomotion(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        동영상 파일에는 IMU/GPS 데이터가 없으므로 Mock egomotion 반환.
        PhysicalAI 데이터셋 사용 시에는 egomotion.parquet에서 읽어야 함.
        """
        self._frame_idx += 1
        t = self._frame_idx * 0.1
        ego_xyz = torch.tensor([t * 8.0, 0.0, 0.0], dtype=torch.float32)
        ego_rot = torch.eye(3, dtype=torch.float32)
        return ego_xyz, ego_rot

    @property
    def is_done(self) -> bool:
        return self._finished


# ─── 모델 래퍼 ────────────────────────────────────────────────────────────────

class AlpamayoStreamingInference:
    """
    Alpamayo 1.5의 공식 API를 래핑한 스트리밍 추론 클래스.
    슬라이딩 윈도우 버퍼에서 model_input을 받아 추론을 실행한다.
    """

    def __init__(self, model_path: str, dtype: str = "bf16", attn: str = "eager", lang: str = "ko"):
        self.lang = lang
        self.model = None
        self.processor = None
        self.helper = None
        self._load_model(model_path, dtype)

    def _load_model(self, model_path: str, dtype: str):
        logger.info(f"모델 로딩: {model_path}")

        # ── Step 1: 패키지 import 확인 ────────────────────────────────────
        try:
            from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
            from alpamayo1_5 import helper as alpamayo_helper
        except ImportError as e:
            logger.warning(f"alpamayo1_5 패키지 없음 ({e}) → Mock 추론 모드.")
            self.model = None
            return

        # ── Step 2: dtype별 로딩 전략 결정 ──────────────────────────────
        #
        # ⚠️ dtype 옵션 설명:
        #
        #   "bf16" (기본, 권장):
        #       from_pretrained(dtype=bfloat16) — 공식 코드와 동일
        #       실측: ~4.8s/inference on Thor
        #
        #   "int4" (bitsandbytes 4비트 양자화):
        #       Weight-only INT4 양자화 (NF4 format)
        #       KV cache는 BF16 유지 (양자화 금지 영역 자동 보호)
        #       설치 필요: pip install bitsandbytes
        #       예상: ~1.5-2.5s/inference (메모리 절반, 속도 향상)
        #
        #   "fp4" 는 현재 지원 불가:
        #       PyTorch에 FP4 storage 타입 없음
        #       Thor Blackwell FP4 (2,070 TFLOPS) 달성은
        #       TensorRT-LLM 엔진 변환 필요 — 별도 작업
        #
        #   ❌ 잘못된 구현 (이전 버전):
        #       "fp4" → torch.float8_e4m3fn 으로 매핑 (FP8이고, from_pretrained
        #       전체 일괄 캐스팅이라 KV cache도 캐스팅됨 — 동작 불가)

        if dtype == "int4":
            # bitsandbytes INT4: weights만 4비트, compute/KV cache는 BF16 유지
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,  # compute는 BF16
                    bnb_4bit_quant_type="nf4",              # NormalFloat4
                    bnb_4bit_use_double_quant=True,         # 이중 양자화로 추가 압축
                )
                logger.info("  INT4 양자화 (bitsandbytes NF4) 적용 중...")
                logger.info("  ※ KV cache는 BF16으로 자동 유지됨")
                self.model = Alpamayo1_5.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                ).eval()  # INT4 모델은 .cuda() 불필요 (자동으로 GPU 배치됨)
            except ImportError:
                logger.error(
                    "bitsandbytes 미설치. INT4 불가.\n"
                    "  python3 -m pip install bitsandbytes\n"
                    "BF16으로 폴백."
                )
                dtype = "bf16"  # 폴백
                self.model = Alpamayo1_5.from_pretrained(
                    model_path, dtype=torch.bfloat16
                ).cuda().eval()

        elif dtype == "fp4":
            # FP4는 현재 PyTorch/HuggingFace로 직접 불가
            # TensorRT-LLM 엔진 변환이 필요 (별도 작업)
            logger.warning(
                "fp4 dtype은 from_pretrained()으로 직접 지원되지 않습니다.\n"
                "  Thor Blackwell FP4 (2,070 TFLOPS) 달성:\n"
                "    → TensorRT-LLM으로 엔진 변환 필요\n"
                "    → 또는 --dtype int4 (bitsandbytes, 즉시 사용 가능)\n"
                "  BF16으로 폴백합니다."
            )
            dtype = "bf16"
            self.model = Alpamayo1_5.from_pretrained(
                model_path, dtype=torch.bfloat16
            ).cuda().eval()

        else:
            # bf16 / fp16 — 공식 코드 방식
            dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
            torch_dtype = dtype_map.get(dtype, torch.bfloat16)
            logger.info(f"  모델 weights 로드 중 (dtype={dtype})...")
            self.model = Alpamayo1_5.from_pretrained(
                model_path,
                dtype=torch_dtype,
            ).cuda().eval()

        try:
            logger.info(f"  모델 weights 로드 ✅")
            # Processor: model.tokenizer에서 생성 (공식 방식)
            logger.info(f"  Processor 생성 중...")
            self.processor = alpamayo_helper.get_processor(self.model.tokenizer)
            self.helper = alpamayo_helper
            logger.info(f"모델 + Processor 로딩 완료 ✅  (dtype={dtype})")

        except Exception as e:
            logger.error(f"  모델 로딩 실패: {type(e).__name__}: {e}")
            logger.warning("Mock 추론 모드로 전환.")
            self.model = None

    def _build_nav_text(self) -> str:
        """언어 설정에 따른 내비게이션/추론 프롬프트."""
        if self.lang == "ko":
            return (
                "현재 주행 상황을 분석하고, 결정한 행동과 그 이유를 한국어로 설명하세요. "
                '형식: {"상황": "...", "판단": "...", "행동": "...", "안전근거": "..."}'
            )
        return "Analyze the current driving situation and output trajectory with chain-of-causation reasoning."

    def infer(self, buffer_input: dict) -> dict:
        """
        슬라이딩 윈도우 버퍼 출력 → 모델 추론 → 결과 반환.

        Args:
            buffer_input: SlidingWindowBuffer.get_model_input() 반환값

        Returns:
            dict:
              pred_xyz: ndarray (64, 3) — 미래 64 웨이포인트
              pred_rot: ndarray (64, 3, 3)
              cot: str — 추론 텍스트 (없으면 "")
              latency_ms: float
        """
        t_start = time.perf_counter()

        if self.model is None:
            # Mock 추론: 실제 모델 없을 때 파이프라인 테스트용
            time.sleep(0.05)  # 50ms 추론 시뮬레이션
            pred_xyz = np.random.randn(64, 3).astype(np.float32)
            pred_xyz[:, 0] = np.cumsum(np.abs(np.random.randn(64)) * 0.5)  # 전진
            return {
                "pred_xyz": pred_xyz,
                "pred_rot": np.tile(np.eye(3), (64, 1, 1)).astype(np.float32),
                "cot": "[MOCK] 직선 주행 유지",
                "latency_ms": (time.perf_counter() - t_start) * 1000,
            }

        # ── 공식 API 호출 (NVlabs/alpamayo1.5/test_inference.py 기준) ──────
        # 1. create_message: 이미지 프레임 → chat messages 구성
        #    공식: helper.create_message(frames=data["image_frames"].flatten(0,1),
        #                               camera_indices=data["camera_indices"])
        frames_flat = buffer_input["image_frames"].flatten(0, 1)
        # shape: (N_cameras × N_frames, 3, H, W) = (16, 3, 320, 576)

        messages = self.helper.create_message(
            frames=frames_flat,
            camera_indices=buffer_input["camera_indices"],
        )

        # 2. 토크나이즈
        #    공식: processor.apply_chat_template(messages, tokenize=True, ...)
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_inputs = {
            "tokenized_data": tokenized,
            "ego_history_xyz": buffer_input["ego_history_xyz"],
            "ego_history_rot": buffer_input["ego_history_rot"],
        }

        # 3. GPU 이동 (공식: helper.to_device(model_inputs, "cuda"))
        model_inputs = self.helper.to_device(model_inputs, "cuda")

        # 4. 추론
        #    공식: torch.autocast("cuda", dtype=torch.bfloat16) 사용
        #    (torch.no_grad()가 아님 — autocast로 bf16 mixed precision 적용)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz_t, pred_rot_t, extra = (
                self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    return_extra=True,
                )
            )

        # pred_xyz_t 실제 shape: (batch=1, seq=1, num_traj=1, steps=64, coords=3)
        # 공식 test_inference.py: pred_xyz.cpu().numpy()[0, 0, :, :, :2]
        #   [0]=batch, [0]=seq, [:]=traj, [:]=steps, [:2]=xy
        # → 우리는 첫 번째 trajectory만 사용: [0, 0, 0]
        pred_xyz = pred_xyz_t[0, 0, 0].cpu().numpy()   # (64, 3) ← steps × coords
        pred_rot = pred_rot_t[0, 0, 0].cpu().numpy()   # (64, 3, 3)

        # cot 추출: extra["cot"]의 타입이 str / list / numpy.ndarray 중 하나
        # 공식 코드: extra["cot"][0] → trajectory별 list or ndarray
        raw_cot = extra.get("cot", "")
        if isinstance(raw_cot, np.ndarray):
            # ndarray of strings → 첫 번째 원소를 Python str으로 변환
            cot = str(raw_cot.flat[0]) if raw_cot.size > 0 else ""
        elif isinstance(raw_cot, (list, tuple)):
            cot = str(raw_cot[0]) if len(raw_cot) > 0 else ""
        else:
            cot = str(raw_cot) if raw_cot else ""

        return {
            "pred_xyz": pred_xyz,       # (64, 3) — 6.4초 미래 궤적
            "pred_rot": pred_rot,       # (64, 3, 3)
            "cot": cot,
            "latency_ms": (time.perf_counter() - t_start) * 1000,
        }


# ─── 메인 스트리밍 루프 ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alpamayo 1.5 실시간 스트리밍 추론")
    p.add_argument("--model",   default="nvidia/Alpamayo-1.5-10B")
    p.add_argument("--source",  default="mock",
                   help="'mock' 또는 카메라 device ID (예: '0,1,2,3')")
    p.add_argument("--hz",      type=float, default=10.0,
                   help="목표 추론 빈도 (Hz). 10=0.1초마다")
    p.add_argument("--dtype",   default="bf16", choices=["fp4", "bf16", "fp16"])
    p.add_argument("--attn",    default="eager", choices=["flash_attn2", "sdpa", "eager"],
                   help="Attention 구현. Alpamayo1_5는 flash_attn2/eager 지원 (sdpa 미지원)."
                        " 기본값: eager (가장 안전)")
    p.add_argument("--lang",    default="ko", choices=["ko", "en"])
    p.add_argument("--bench",   action="store_true",
                   help="벤치마크 모드: latency 측정 후 종료")
    p.add_argument("--iterations", type=int, default=50,
                   help="벤치마크 반복 횟수")
    p.add_argument("--output_dir", default="evaluation/results/streaming/")
    p.add_argument("--save_traces", action="store_true")
    p.add_argument("--skip-model", action="store_true", dest="skip_model",
                   help="모델 로딩 없이 Mock 추론으로 파이프라인만 테스트."
                        " --source mock --bench 조합 시 추천 (22GB 다운로드 생략).")
    return p.parse_args()


def run_capture_thread(
    camera: MockCameraSource,
    buf: SlidingWindowBuffer,
    stop_event: threading.Event,
    hz: float,
):
    """
    별도 스레드에서 카메라 캡처 → 버퍼 push.
    추론 스레드와 병렬 실행된다.
    """
    interval = 1.0 / hz
    logger.info(f"캡처 스레드 시작: {hz} Hz (간격 {interval*1000:.0f}ms)")

    while not stop_event.is_set():
        t_cap = time.perf_counter()

        frames = camera.get_frames()              # (4, 3, 320, 576)
        ego_xyz, ego_rot = camera.get_egomotion() # (3,), (3, 3)
        ts_us = int(time.time() * 1e6)

        buf.push_frame(frames, ego_xyz, ego_rot, ts_us)

        # 다음 캡처 시각까지 대기
        elapsed = time.perf_counter() - t_cap
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def streaming_loop(args: argparse.Namespace):
    """메인 추론 루프."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 컴포넌트 초기화 ────────────────────────────────────────────────────
    if args.source == "mock":
        camera = MockCameraSource()
    elif args.source.endswith((".mp4", ".avi", ".mov", ".mkv")) or \
         any(p.endswith((".mp4", ".avi", ".mov", ".mkv")) for p in args.source.split(",")):
        # 동영상 파일 모드 (단일 또는 카메라별 4개)
        video_paths = args.source.split(",")
        camera = VideoFileCameraSource(video_paths, target_hz=args.hz)
        logger.info(f"동영상 파일 소스: {video_paths}")
    else:
        device_ids = [int(x) for x in args.source.split(",")]
        camera = RawResizeCameraSource(device_ids)

    buf = SlidingWindowBuffer()

    if args.skip_model:
        # 파이프라인만 테스트 — 모델 로딩 없이 Mock 추론 사용
        logger.info("--skip-model: 모델 로딩 생략, Mock 추론 모드.")
        inferencer = AlpamayoStreamingInference.__new__(AlpamayoStreamingInference)
        inferencer.lang = args.lang
        inferencer.model = None     # infer() 내부에서 None 체크 → Mock 동작
        inferencer.processor = None
        inferencer.helper = None
    else:
        inferencer = AlpamayoStreamingInference(
            model_path=args.model,
            dtype=args.dtype,
            attn=args.attn,
            lang=args.lang,
        )

    # ── 캡처 스레드 시작 ──────────────────────────────────────────────────
    stop_event = threading.Event()
    cap_thread = threading.Thread(
        target=run_capture_thread,
        args=(camera, buf, stop_event, args.hz),
        daemon=True,
        name="CaptureThread",
    )
    cap_thread.start()

    # ── 버퍼 초기 충전 대기 ──────────────────────────────────────────────
    logger.info("버퍼 초기 충전 중...")
    warmup_needed = max(N_FRAMES, N_EGO_HISTORY)  # 16 프레임
    warmup_time = warmup_needed / args.hz
    t_wait = time.perf_counter()
    while not buf.is_ready:
        time.sleep(0.05)
        if time.perf_counter() - t_wait > warmup_time + 2.0:
            logger.error("버퍼 초기화 타임아웃!")
            stop_event.set()
            return
    logger.info("버퍼 준비 완료. 추론 시작.")

    # ── 추론 루프 ─────────────────────────────────────────────────────────
    latencies: list[float] = []
    trace_log: list[dict] = []
    interval = 1.0 / args.hz
    iteration = 0
    max_iter = args.iterations if args.bench else None

    logger.info(f"{'벤치마크' if args.bench else '실시간 스트리밍'} 모드 시작. "
                f"Ctrl+C로 중단.")

    try:
        while True:
            t_loop = time.perf_counter()
            iteration += 1

            # 슬라이딩 윈도우에서 모델 입력 구성
            model_input = buf.get_model_input()
            if model_input is None:
                logger.warning("버퍼 데이터 부족 — 스킵")
                time.sleep(0.01)
                continue

            # 추론 실행
            result = inferencer.infer(model_input)
            lat = result["latency_ms"]
            latencies.append(lat)

            # 로그 출력
            pred = result["pred_xyz"]   # (64, 3)
            wp0 = pred[0]               # 0.1초 후 예측 위치
            wp9 = pred[9]               # 1.0초 후 예측 위치
            # 실제 모델 latency는 수 초 → 100ms 기준 대신 hz 목표 대비 표시
            target_ms = 1000.0 / args.hz
            status = "✅" if lat <= target_ms else f"⏱ {lat/1000:.1f}s"
            logger.info(
                f"[{iteration:04d}] {status} "
                f"latency={lat:.0f}ms ({lat/1000:.2f}s) | "
                f"wp[0]=({wp0[0]:.2f},{wp0[1]:.2f})m "
                f"wp[9]=({wp9[0]:.2f},{wp9[1]:.2f})m"
            )

            # CoT (추론 텍스트) 매번 출력 (실제 모델일 때 중요)
            cot_text = result["cot"]  # infer()에서 이미 str로 변환됨
            if cot_text:
                cot_preview = cot_text[:300].replace("\n", " ")
                logger.info(f"  CoT: {cot_preview}")

            # 100회마다 통계
            if iteration % 100 == 0 and len(latencies) >= 100:
                recent = latencies[-100:]
                logger.info(
                    f"  [통계] avg={np.mean(recent):.1f}ms "
                    f"p50={np.percentile(recent,50):.1f}ms "
                    f"p95={np.percentile(recent,95):.1f}ms"
                )

            if args.save_traces:
                trace_log.append({
                    "iteration": iteration,
                    "latency_ms": round(lat, 2),
                    "t0_us": model_input["t0_us"],
                    "waypoints_5": result["pred_xyz"][:5].tolist(),
                    "cot": result["cot"][:300] if result["cot"] else "",
                })

            # 벤치마크 종료 조건
            if max_iter and iteration >= max_iter:
                logger.info(f"벤치마크 {max_iter}회 완료.")
                break

            # 다음 추론 시각까지 대기 (실시간 모드)
            if not args.bench:
                elapsed = time.perf_counter() - t_loop
                sleep_t = interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
                elif elapsed > interval * 1.5:
                    logger.warning(
                        f"  추론({lat:.0f}ms) > 목표 인터벌({interval*1000:.0f}ms). "
                        f"Hz를 낮추거나 FP4 dtype을 사용하세요."
                    )

    except KeyboardInterrupt:
        logger.info("사용자 중단.")

    finally:
        stop_event.set()

        # ── 결과 저장 ──────────────────────────────────────────────────────
        if latencies:
            summary = {
                "mode": "benchmark" if args.bench else "streaming",
                "source": args.source,
                "dtype": args.dtype,
                "hz_target": args.hz,
                "total_frames": iteration,
                "avg_latency_ms": round(float(np.mean(latencies)), 2),
                "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
                "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
                "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
                "target_interval_ms": 1000.0 / args.hz,
                "frames_within_target_pct": round(
                    100 * sum(l <= 1000.0 / args.hz for l in latencies) / len(latencies), 1
                ),
            }
            summary_path = output_dir / "streaming_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info("=" * 55)
            logger.info(f"  총 {iteration}프레임 처리")
            logger.info(f"  평균 latency:  {summary['avg_latency_ms']} ms")
            logger.info(f"  P95 latency:   {summary['p95_latency_ms']} ms")
            logger.info(f"  목표 내 비율:  {summary['frames_within_target_pct']}%")
            logger.info(f"  결과: {summary_path}")
            logger.info("=" * 55)

        if args.save_traces and trace_log:
            trace_path = output_dir / "streaming_traces.jsonl"
            with open(trace_path, "w", encoding="utf-8") as f:
                for entry in trace_log:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(f"추론 로그: {trace_path}")


def main():
    args = parse_args()

    logger.info("=" * 55)
    logger.info("  Alpamayo 1.5 — 실시간 스트리밍 추론")
    logger.info("=" * 55)
    logger.info(f"  입력 소스:  {args.source}")
    logger.info(f"  추론 빈도:  {args.hz} Hz ({1000/args.hz:.0f}ms 목표)")
    logger.info(f"  dtype:      {args.dtype}")
    logger.info(f"  언어:       {'한국어' if args.lang == 'ko' else 'English'}")
    logger.info(f"  이미지:     {N_CAMERAS}카메라 × {N_FRAMES}프레임 ({IMG_H}×{IMG_W})")
    logger.info(f"  Ego 이력:   {N_EGO_HISTORY} steps ({N_EGO_HISTORY/FRAME_HZ:.1f}초)")
    logger.info(f"  미래 예측:  {N_FUTURE_STEPS} waypoints ({N_FUTURE_STEPS/FRAME_HZ:.1f}초)")
    if torch.cuda.is_available():
        logger.info(f"  GPU:        {torch.cuda.get_device_name(0)}")
    logger.info("=" * 55)

    streaming_loop(args)


if __name__ == "__main__":
    main()
