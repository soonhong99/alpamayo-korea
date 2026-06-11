"""
exp1_decode_skip.py — 실험 1: Decode Skip / Adaptive Decode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목적:
  Chain-of-Causation(CoC) 토큰 수를 0~16 사이에서 제어하여
  "얼마나 생성해야 궤적 품질이 유지되는가"의 하한을 실측한다.

  max_coc_tokens 값을 파라미터로 제어 (patch_alpamayo_coc.py 패치 필수):
    0   → 즉시 EOS 강제 → Prefill 마지막 hidden state만으로 Flow 실행 (Decode Skip)
    1~13 → N개 CoC 토큰 후 EOS 강제 삽입 → 단축 CoC
    16  → 기준선(baseline, 실측 기준)

  패치 방식:
    ForceEarlyEOS LogitsProcessor가 N번째 스텝 이후 <|traj_future_start|>(EOS)를 강제.
    → StopAfterEOS가 EOS 감지 → +1 토큰 후 생성 종료 → Flow Matching에 b_star hidden 전달.

평가 지표:
  - ADE (Average Displacement Error): N=16 baseline 대비 64개 waypoint 평균 L2 거리
  - FDE (Final Displacement Error):   waypoint[63] 단독 L2 거리
  - wp[0] 물리 검증: x ∈ [0.5, 1.5]m, |y| < 0.3m
  - CoC 품질: 완결성, 행동 키워드, 최소 길이
  - Total latency: mean ± std, P95 (CUDA Event 기반)

합격 기준:
  ADE < 0.2m  AND  FDE < 1.0m  AND  wp[0].x ∈ [0.5, 1.5]m

사용법:
  # 모델 로드 후 실행 (Thor에서)
  python scripts/exp1_decode_skip.py \
      --model ~/alpamayo1.5/checkpoints/alpamayo_base \
      --runs 10 --dtype bf16

  # Mock 모드 (파이프라인 테스트만)
  python scripts/exp1_decode_skip.py --mock

사전 조건:
  Thor에서 반드시 아래 패치를 먼저 실행해야 한다:
    python3 ~/alpamayo1.5/scripts/patch_alpamayo_coc.py
  패치가 없으면 max_coc_tokens 파라미터가 무시되어 기존 max_generation_length 동작과 동일함.

출력:
  evaluation/results/streaming/exp1_decode_skip/
  ├── baseline_waypoints.npy      ← N=16 기준 궤적 (ADE/FDE 계산 기준)
  ├── tokens_0.json
  ├── tokens_1.json
  ├── ...
  ├── tokens_16.json
  └── summary.json                ← 전체 sweep 요약
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ── 상수 ────────────────────────────────────────────────────────────────────────
N_CAMERAS      = 4
N_FRAMES       = 4
N_EGO_HISTORY  = 16
N_FUTURE_STEPS = 64
IMG_H, IMG_W   = 320, 576

# 실험 계획서에 명시된 sweep 조건
# N=0: ForceEarlyEOS가 즉시 EOS 강제 → max_new_tokens=0+4=4로 설정 (patch 내부에서 처리)
#       transformers generate()의 max_new_tokens=0 거부 문제를 ForceEarlyEOS가 우회.
# N=1~13: N개 CoC 후 EOS 강제 → 실제 CoC 토큰 수를 정확히 제어.
# N=16: baseline (EOS를 강제하지 않고 자연 생성)
TOKEN_SWEEP = [0, 1, 3, 5, 8, 10, 13, 16]

# 품질 판정 기준 (실험계획서 §품질 평가 기준)
ADE_PASS   = 0.2   # m
ADE_MARGIN = 0.5   # m
FDE_PASS   = 1.0   # m
FDE_MARGIN = 3.0   # m
WP0_X_MIN, WP0_X_MAX = 0.5, 1.5   # m
WP0_Y_ABS_MAX         = 0.3        # m
TRAJ_CONT_PASS        = 2.0        # m (연속 waypoint 간 최대 거리)

ACTION_KEYWORDS = [
    "keep", "decelerate", "turn", "stop", "merge",
    "유지", "감속", "선회", "정지", "합류", "직진", "차선",
]

# ── CUDA 타이머 ──────────────────────────────────────────────────────────────────

class CUDATimer:
    """torch.cuda.Event 기반 GPU 타이머."""

    def __init__(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)
        self._valid = False

    def start(self):
        torch.cuda.synchronize()
        self._s.record()
        self._valid = False

    def stop(self) -> float:
        """stop 후 경과 시간(ms) 반환."""
        self._e.record()
        torch.cuda.synchronize()
        self._valid = True
        return self._s.elapsed_time(self._e)

    def elapsed_ms(self) -> float:
        return self._s.elapsed_time(self._e) if self._valid else 0.0


# ── 입력 생성 ────────────────────────────────────────────────────────────────────

# 기존 프로파일링(260513_profile_v4.py)에서 사용한 PhysicalAI 클립 ID
# 이 클립으로 Vision=714ms, Prefill=1472ms, Decode=1926ms, Flow=890ms 확정됨
PROFILING_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
PROFILING_T0_US   = 5_100_000

def load_physicalai_input() -> dict:
    """
    기존 프로파일링과 동일한 PhysicalAI 클립을 로드한다.

    왜 이 입력을 쓰는가:
      - 기존 profiling 5,009ms baseline이 이 클립으로 측정됨
      - max_generation_length sweep 결과를 baseline 5,009ms와 직접 비교 가능
      - 실제 주행 장면 → CoC 일관성 있음, ADE/FDE 비교가 의미 있음
      - random noise 입력(이전 버전)은 모델이 없는 장면을 hallucinate → 비교 불가

    Fallback:
      물리적AI 라이브러리가 없으면 mock_input()으로 폴백한다.
    """
    try:
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        logger.info(f"PhysicalAI 클립 로드: {PROFILING_CLIP_ID}  t0={PROFILING_T0_US}us")
        data = load_physical_aiavdataset(PROFILING_CLIP_ID, t0_us=PROFILING_T0_US)
        logger.info("입력 로드 완료 (실제 주행 데이터)")
        return data
    except Exception as e:
        logger.warning(f"PhysicalAI 로드 실패 ({e}) → mock 입력으로 폴백")
        return _make_mock_input()


def _make_mock_input() -> dict:
    """
    PhysicalAI 데이터셋을 사용할 수 없을 때 폴백용 mock 입력.
    이 입력으로는 CoC가 랜덤하게 변하고 latency도 실제와 다를 수 있음.
    실험 결과를 baseline 5,009ms와 비교하려면 반드시 PhysicalAI 입력 사용 필요.
    """
    logger.warning("mock 입력 사용 중 — 실험 결과의 절대값은 신뢰 불가 (비교만 가능)")

    rng = torch.Generator()
    rng.manual_seed(42)

    image_frames = torch.sigmoid(
        torch.randn(N_CAMERAS, N_FRAMES, 3, IMG_H, IMG_W, generator=rng, dtype=torch.float32)
    )
    ego_history_xyz = torch.zeros(1, 1, N_EGO_HISTORY, 3)
    ego_history_xyz[..., 0] = torch.linspace(0.0, (N_EGO_HISTORY - 1) * 0.8, N_EGO_HISTORY)
    ego_history_rot = torch.zeros(1, 1, N_EGO_HISTORY, 3, 3)
    ego_history_rot[..., 0, 0] = 1.0
    ego_history_rot[..., 1, 1] = 1.0
    ego_history_rot[..., 2, 2] = 1.0
    camera_indices = torch.arange(N_CAMERAS, dtype=torch.int64)
    relative_timestamps = torch.zeros(N_CAMERAS, N_FRAMES)
    for fi in range(N_FRAMES):
        relative_timestamps[:, fi] = (fi - (N_FRAMES - 1)) * 0.1

    return {
        "image_frames":        image_frames,
        "camera_indices":      camera_indices,
        "ego_history_xyz":     ego_history_xyz,
        "ego_history_rot":     ego_history_rot,
        "relative_timestamps": relative_timestamps,
    }


# ── 모델 로딩 ────────────────────────────────────────────────────────────────────

def load_model(model_path: str, dtype: str):
    """
    Alpamayo 1.5 model + helper load.

    Loading rules (confirmed 2026-05-21):
      Alpamayo1_5.from_pretrained() internally calls hf_hub_download(), which
      rejects absolute local paths as repo_id → HFValidationError.
      Must use HF repo ID format ("nvidia/Alpamayo-1.5-10B").

    Supported cases:
      Case A — HF default cache (~/.cache/huggingface/hub/):
        Pass "nvidia/Alpamayo-1.5-10B" or any HF repo ID.
        HF will find the cached model automatically.
        local_files_only=True prevents any network call.

      Case B — Custom local directory (e.g. ~/alpamayo1.5/checkpoints/alpamayo_base):
        Set HF_HUB_CACHE env var to point to the directory that CONTAINS
        the "models--nvidia--Alpamayo-1.5-10B" folder, then use the repo ID.
        OR: set --model to "nvidia/Alpamayo-1.5-10B" directly if model is in HF cache.

    To find the model on Thor, run:
      find ~/.cache/huggingface -name "config.json" 2>/dev/null | grep -i alpamayo
      ls ~/alpamayo1.5/checkpoints/
    """
    import os

    try:
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        from alpamayo1_5 import helper as alpamayo_helper

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # ── Resolve pretrained_id ────────────────────────────────────────────
        # Alpamayo1_5.from_pretrained() rejects absolute local paths.
        # If a local path is given, detect it and handle accordingly.
        p = Path(model_path).expanduser().resolve()
        if p.exists() and p.is_dir():
            # Local directory detected.
            # Check if it looks like an HF hub cache root
            # (contains "models--nvidia--Alpamayo-1.5-10B" subdir)
            hf_subdir = p / "models--nvidia--Alpamayo-1.5-10B"
            if hf_subdir.exists():
                # p is an HF hub cache root → set HF_HUB_CACHE and use repo ID
                os.environ["HF_HUB_CACHE"] = str(p)
                pretrained_id = "nvidia/Alpamayo-1.5-10B"
                logger.info(f"HF hub cache root detected: {p}")
                logger.info(f"  HF_HUB_CACHE={p}  pretrained_id={pretrained_id}")
            else:
                # Plain local snapshot directory (e.g. checkpoints/alpamayo_base/)
                # Alpamayo1_5.from_pretrained() cannot load this directly.
                # Raise a clear error instead of an obscure HFValidationError.
                raise ValueError(
                    f"\n\n[load_model] Local path detected but NOT in HF hub cache format:\n"
                    f"  path: {p}\n"
                    f"  Alpamayo1_5.from_pretrained() requires either:\n"
                    f"    (A) HF repo ID:  --model nvidia/Alpamayo-1.5-10B\n"
                    f"        (model must be in ~/.cache/huggingface/hub/)\n"
                    f"    (B) HF cache root dir:  --model /path/that/contains/models--nvidia--Alpamayo-1.5-10B/\n"
                    f"  To check model location on Thor:\n"
                    f"    find ~/.cache/huggingface -name 'config.json' | grep -i alpamayo\n"
                    f"    ls ~/alpamayo1.5/checkpoints/\n"
                )
        else:
            # Treat as HF repo ID (e.g. "nvidia/Alpamayo-1.5-10B")
            pretrained_id = model_path

        logger.info(f"Loading model: {pretrained_id}  (dtype={dtype})")
        logger.info("  local_files_only=True — network calls blocked, using cached weights.")
        model = Alpamayo1_5.from_pretrained(
            pretrained_id,
            dtype=torch_dtype,
            attn_implementation="eager",   # Thor: sdpa not supported (confirmed 2026-05-19)
            local_files_only=True,         # Never download; fail fast if not cached
        ).cuda().eval()

        processor = alpamayo_helper.get_processor(model.tokenizer)
        logger.info("Model loaded successfully.")
        return model, processor, alpamayo_helper

    except ImportError as e:
        logger.error(f"alpamayo1_5 package not found: {e}")
        raise


# ── 단일 추론 실행 ────────────────────────────────────────────────────────────────

def run_single_inference(
    model,
    processor,
    helper,
    inp: dict,
    max_coc_tokens: int,
    seed: int = 0,
) -> dict:
    """
    주어진 max_coc_tokens로 추론 1회 실행.

    patch_alpamayo_coc.py 패치 후 ForceEarlyEOS가 정확히 max_coc_tokens개 CoC 후
    <|traj_future_start|> EOS를 강제 삽입한다.
      max_coc_tokens=0:  즉시 EOS → Prefill hidden state만으로 Flow 실행
      max_coc_tokens=N:  N개 CoC 후 EOS 강제

    Returns:
        dict:
          waypoints:    ndarray (64, 3)
          cot:          str
          latency_ms:   float (CUDA Event 기반)
          token_count:  int  (실제 생성된 토큰 수)
    """
    # ── 입력 구성 (공식 API 패턴) ──────────────────────────────────────────────
    frames_flat = inp["image_frames"].flatten(0, 1)   # (16, 3, H, W)

    messages = helper.create_message(
        frames=frames_flat,
        camera_indices=inp["camera_indices"],
    )

    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = {
        "tokenized_data":    tokenized,
        "ego_history_xyz":   inp["ego_history_xyz"],
        "ego_history_rot":   inp["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    # ── 추론 (CUDA Event 타이밍) ───────────────────────────────────────────────
    timer = CUDATimer()

    # 재현성: decode의 샘플링 시드 고정
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        timer.start()
        pred_xyz_t, pred_rot_t, extra = (
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=max_coc_tokens + 8,  # 안전망: CoC + EOS + 여유
                max_coc_tokens=max_coc_tokens,             # ForceEarlyEOS 실제 제어값
                return_extra=True,
            )
        )
        latency_ms = timer.stop()

    # ── 결과 추출 ──────────────────────────────────────────────────────────────
    waypoints = pred_xyz_t[0, 0, 0].cpu().float().numpy()   # (64, 3)

    raw_cot = extra.get("cot", "")
    if isinstance(raw_cot, np.ndarray):
        cot = str(raw_cot.flat[0]) if raw_cot.size > 0 else ""
    elif isinstance(raw_cot, (list, tuple)):
        cot = str(raw_cot[0]) if len(raw_cot) > 0 else ""
    else:
        cot = str(raw_cot) if raw_cot else ""

    # 실제 생성된 토큰 수 추출
    # 우선순위: ① extra에 token_ids 있으면 직접 카운트
    #           ② tokenizer로 CoC 텍스트를 역산
    #           ③ 단어 수로 근사 (fallback)
    token_count = 0
    try:
        token_ids = extra.get("generated_token_ids", None)
        if token_ids is not None:
            if isinstance(token_ids, torch.Tensor):
                token_count = int(token_ids.shape[-1])
            elif isinstance(token_ids, (list, np.ndarray)):
                token_count = len(token_ids)
        elif cot:
            # tokenizer로 역산 (EOS 제외)
            enc = processor.tokenizer(cot, return_tensors="pt")
            token_count = int(enc["input_ids"].shape[-1])
    except Exception:
        # fallback: 단어 수 (단어 ≠ 토큰이나 근사값으로 사용)
        token_count = len(cot.split()) if cot else 0

    return {
        "waypoints":   waypoints,
        "cot":         cot,
        "latency_ms":  latency_ms,
        "token_count": token_count,
    }


# ── Mock 추론 (모델 없이 파이프라인 테스트용) ────────────────────────────────────

def run_mock_inference(max_coc_tokens: int) -> dict:
    """
    실제 모델 없이 Mock 궤적 반환.
    latency는 max_coc_tokens에 비례하는 추정값 사용.

    기준:
      - 57ms/token (NSight GPU 커널 실측)
      - 53ms/token 오버헤드
      - 나머지 Phase: 3,077ms (Vision + Prefill + Flow)
    """
    base_ms = 3_077.0
    per_token_ms = 110.0

    simulated_ms = base_ms + max_coc_tokens * per_token_ms
    # 약간의 노이즈 추가
    simulated_ms += np.random.normal(0, 50)
    time.sleep(simulated_ms / 1000.0)

    # 물리적으로 합리적인 Mock 궤적 (직선 주행)
    waypoints = np.zeros((N_FUTURE_STEPS, 3), dtype=np.float32)
    for i in range(N_FUTURE_STEPS):
        t = (i + 1) * 0.1   # 0.1초 간격
        waypoints[i, 0] = 8.0 * t     # 직진 (8 m/s)
        waypoints[i, 1] = 0.0
        waypoints[i, 2] = 0.0

    # max_coc_tokens=0이면 궤적을 약간 degradate (테스트용)
    if max_coc_tokens == 0:
        waypoints += np.random.normal(0, 0.05, waypoints.shape).astype(np.float32)

    mock_keywords = ["Keep", "유지", "직진"]
    cot = ""
    if max_coc_tokens > 0:
        cot = f"[MOCK] {mock_keywords[max_coc_tokens % len(mock_keywords)]} lane. "
        cot *= min(max_coc_tokens, 5)

    return {
        "waypoints":   waypoints,
        "cot":         cot,
        "latency_ms":  simulated_ms,
        "token_count": max_coc_tokens,
    }


# ── 품질 평가 ────────────────────────────────────────────────────────────────────

def compute_ade(pred: np.ndarray, ref: np.ndarray) -> float:
    """Average Displacement Error (m): 64개 waypoint 평균 L2 거리."""
    diff = pred[:, :2] - ref[:, :2]   # xy만 사용
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def compute_fde(pred: np.ndarray, ref: np.ndarray) -> float:
    """Final Displacement Error (m): waypoint[63] 단독 L2 거리."""
    diff = pred[63, :2] - ref[63, :2]
    return float(np.linalg.norm(diff))


def check_wp0_physics(waypoints: np.ndarray) -> dict:
    """
    첫 번째 waypoint의 물리적 합리성 검증.

    기준:
      wp[0].x ∈ [0.5, 1.5]m  (0.1초 후 전진 거리)
      |wp[0].y| < 0.3m       (횡방향 오차 허용범위)
    """
    x = float(waypoints[0, 0])
    y = float(waypoints[0, 1])
    x_ok = WP0_X_MIN <= x <= WP0_X_MAX
    y_ok = abs(y) < WP0_Y_ABS_MAX
    return {
        "x":    round(x, 4),
        "y":    round(y, 4),
        "x_ok": x_ok,
        "y_ok": y_ok,
        "pass": x_ok and y_ok,
    }


def check_trajectory_continuity(waypoints: np.ndarray) -> dict:
    """연속 waypoint 간 최대 이동 거리 검증."""
    diffs = np.linalg.norm(np.diff(waypoints[:, :2], axis=0), axis=1)
    max_jump = float(np.max(diffs))
    return {
        "max_jump_m": round(max_jump, 4),
        "pass": max_jump < TRAJ_CONT_PASS,
    }


def evaluate_coc_quality(cot: str, max_coc: int) -> dict:
    """CoC 텍스트 품질 평가 (Decode 관련 실험에서만 의미 있음)."""
    if max_coc == 0:
        # 0 token → CoC 없음이 정상
        return {
            "length": 0,
            "complete": True,    # 0 token은 CoC 없는 게 의도된 동작
            "has_keyword": None,
            "verdict": "N/A (Decode Skip)",
        }

    length = len(cot.strip())
    has_keyword = any(kw.lower() in cot.lower() for kw in ACTION_KEYWORDS)
    complete = length >= 15 and (cot.strip().endswith((".", ".", "！", "!", "다", "요")))
    # max_coc가 작으면 CoC 텍스트가 짧은 게 정상 — 길이 기준을 비례 완화
    min_len = max(5, min(15, max_coc * 3))
    verdict = "pass" if (complete and has_keyword and length >= min_len) else "fail"

    return {
        "length":      length,
        "complete":    complete,
        "has_keyword": has_keyword,
        "verdict":     verdict,
    }


def judge_quality(ade: Optional[float], fde: Optional[float], wp0: dict) -> str:
    """
    ADE/FDE + wp[0] 물리 기준으로 최종 품질 판정.

    ADE/FDE가 None이면 baseline 본인이라 비교 불가 → "baseline"
    """
    if ade is None:
        return "baseline"

    ade_ok = ade < ADE_PASS
    fde_ok = fde < FDE_PASS
    wp0_ok = wp0["pass"]

    if ade_ok and fde_ok and wp0_ok:
        return "PASS"
    elif ade < ADE_MARGIN and fde < FDE_MARGIN:
        return "MARGINAL"
    else:
        return "FAIL"


# ── 단일 조건 실험 실행 ─────────────────────────────────────────────────────────

def run_condition(
    max_gen: int,
    n_runs: int,
    model,
    processor,
    helper,
    fixed_input: dict,
    baseline_waypoints: Optional[np.ndarray],
    use_mock: bool,
    out_dir: Path,
) -> dict:
    """
    하나의 max_coc_tokens 조건에서 n_runs회 추론을 실행한다.

    Returns:
        condition 결과 dict (JSON 직렬화 가능)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  조건: max_coc_tokens = {max_gen}")
    logger.info(f"  실행 횟수: {n_runs}")
    logger.info(f"{'='*60}")

    latencies = []
    all_waypoints = []
    all_cot = []
    token_counts = []

    for run_idx in range(n_runs):
        logger.info(f"  [{run_idx+1:02d}/{n_runs}] 추론 실행 중...")

        if use_mock:
            result = run_mock_inference(max_gen)
        else:
            result = run_single_inference(
                model, processor, helper,
                fixed_input,
                max_coc_tokens=max_gen,
                seed=run_idx,   # run마다 seed 변경 (sampling variance 포착)
            )

        latencies.append(result["latency_ms"])
        all_waypoints.append(result["waypoints"])
        all_cot.append(result["cot"])
        token_counts.append(result["token_count"])

        logger.info(
            f"  ✓ latency={result['latency_ms']:.1f}ms "
            f"| wp[0]=({result['waypoints'][0,0]:.3f}, {result['waypoints'][0,1]:.3f})m "
            f"| tokens={result['token_count']}"
        )
        if result["cot"]:
            logger.info(f"  CoT: {result['cot'][:120]}")

    # ── 통계 계산 ─────────────────────────────────────────────────────────────
    lat_arr = np.array(latencies)
    wp_arr  = np.array(all_waypoints)   # (n_runs, 64, 3)

    # 대표 궤적: median of runs (L2 기준으로 중앙값에 가장 가까운 run 선택)
    mean_wp = np.mean(wp_arr, axis=0)   # (64, 3)
    dists_to_mean = [compute_ade(w, mean_wp) for w in wp_arr]
    rep_idx = int(np.argmin(dists_to_mean))
    representative_wp = wp_arr[rep_idx]

    # ADE/FDE vs baseline
    if baseline_waypoints is not None:
        ades = [compute_ade(w, baseline_waypoints) for w in wp_arr]
        fdes = [compute_fde(w, baseline_waypoints) for w in wp_arr]
        mean_ade = float(np.mean(ades))
        mean_fde = float(np.mean(fdes))
        std_ade  = float(np.std(ades))
        std_fde  = float(np.std(fdes))
    else:
        # baseline 자신
        ades, fdes = None, None
        mean_ade = mean_fde = std_ade = std_fde = None

    # wp[0] 물리 검증 (대표 run 기준)
    wp0_check = check_wp0_physics(representative_wp)
    cont_check = check_trajectory_continuity(representative_wp)

    # CoC 품질 (대표 run 기준)
    coc_check = evaluate_coc_quality(all_cot[rep_idx], max_coc=max_gen)

    # 최종 판정
    verdict = judge_quality(mean_ade, mean_fde, wp0_check)

    # ── 로그 출력 ──────────────────────────────────────────────────────────────
    logger.info(f"\n  [결과] max_gen={max_gen}")
    logger.info(f"  latency: mean={lat_arr.mean():.1f}ms  std={lat_arr.std():.1f}ms  "
                f"P95={np.percentile(lat_arr, 95):.1f}ms")
    if mean_ade is not None:
        logger.info(f"  ADE:     mean={mean_ade:.4f}m  std={std_ade:.4f}m")
        logger.info(f"  FDE:     mean={mean_fde:.4f}m  std={std_fde:.4f}m")
    logger.info(f"  wp[0]:   x={wp0_check['x']:.3f}m  y={wp0_check['y']:.3f}m  "
                f"{'✅' if wp0_check['pass'] else '❌'}")
    logger.info(f"  cont:    max_jump={cont_check['max_jump_m']:.3f}m  "
                f"{'✅' if cont_check['pass'] else '❌'}")
    logger.info(f"  CoC:     {coc_check['verdict']}  (len={coc_check['length']})")
    logger.info(f"  판정:    {verdict}")

    # ── 결과 저장 ──────────────────────────────────────────────────────────────
    cond_result = {
        "max_coc_tokens":        max_gen,
        "n_runs":                n_runs,
        "verdict":               verdict,
        "latency": {
            "mean_ms":  round(float(lat_arr.mean()), 2),
            "std_ms":   round(float(lat_arr.std()), 2),
            "p50_ms":   round(float(np.percentile(lat_arr, 50)), 2),
            "p95_ms":   round(float(np.percentile(lat_arr, 95)), 2),
            "all_ms":   [round(float(x), 2) for x in latencies],
        },
        "ade": {
            "mean_m":  round(mean_ade, 4) if mean_ade is not None else None,
            "std_m":   round(std_ade, 4) if std_ade is not None else None,
            "all_m":   [round(float(x), 4) for x in ades] if ades is not None else None,
            "pass":    (mean_ade < ADE_PASS) if mean_ade is not None else None,
        },
        "fde": {
            "mean_m":  round(mean_fde, 4) if mean_fde is not None else None,
            "std_m":   round(std_fde, 4) if std_fde is not None else None,
            "all_m":   [round(float(x), 4) for x in fdes] if fdes is not None else None,
            "pass":    (mean_fde < FDE_PASS) if mean_fde is not None else None,
        },
        "wp0_physics": wp0_check,
        "trajectory_continuity": cont_check,
        "coc_quality":           coc_check,
        "token_counts":          token_counts,
        "representative_run_idx": rep_idx,
    }

    out_path = out_dir / f"tokens_{max_gen}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cond_result, f, indent=2, ensure_ascii=False)
    logger.info(f"  → 저장: {out_path}")

    # 대표 궤적도 저장
    np.save(out_dir / f"waypoints_{max_gen}.npy", representative_wp)

    return cond_result


# ── 요약 출력 및 저장 ────────────────────────────────────────────────────────────

def print_summary_table(all_results: list[dict]) -> None:
    """전체 sweep 결과 요약 테이블 출력."""
    header = f"{'coc_tok':>8} | {'latency(ms)':>14} | {'ADE(m)':>10} | {'FDE(m)':>10} | {'wp[0]':>7} | {'판정':>8}"
    logger.info("\n" + "=" * 75)
    logger.info("  실험 1: Decode Skip/Adaptive Decode — 결과 요약")
    logger.info("=" * 75)
    logger.info(header)
    logger.info("-" * 75)

    for r in all_results:
        mg      = r["max_coc_tokens"]
        lat     = r["latency"]["mean_ms"]
        lat_std = r["latency"]["std_ms"]
        ade_v   = r["ade"]["mean_m"]
        fde_v   = r["fde"]["mean_m"]
        wp0_ok  = r["wp0_physics"]["pass"]
        verdict = r["verdict"]

        ade_s   = f"{ade_v:.4f}" if ade_v is not None else "  (base)"
        fde_s   = f"{fde_v:.4f}" if fde_v is not None else "  (base)"
        wp0_s   = "✅" if wp0_ok else "❌"

        verdict_display = {
            "PASS":     "✅ PASS",
            "MARGINAL": "⚠️  MARGINAL",
            "FAIL":     "❌ FAIL",
            "baseline": "— baseline",
        }.get(verdict, verdict)

        logger.info(
            f"{mg:>8} | {lat:>8.1f}±{lat_std:<4.1f} | "
            f"{ade_s:>10} | {fde_s:>10} | {wp0_s:>7} | {verdict_display}"
        )

    logger.info("=" * 75)

    # 품질 유지 하한 판별
    pass_tokens = [
        r["max_coc_tokens"]
        for r in all_results
        if r["verdict"] == "PASS"
    ]
    if pass_tokens:
        min_pass = min(pass_tokens)
        baseline_lat = next(
            (r["latency"]["mean_ms"] for r in all_results if r["max_coc_tokens"] == 16),
            None,
        )
        cond_lat = next(
            (r["latency"]["mean_ms"] for r in all_results if r["max_coc_tokens"] == min_pass),
            None,
        )
        if baseline_lat and cond_lat:
            improvement = baseline_lat / cond_lat
            logger.info(f"\n  품질 유지 최소 토큰 수: {min_pass}")
            logger.info(f"  baseline 대비 latency 개선: {improvement:.2f}×")
            logger.info(f"  (baseline={baseline_lat:.0f}ms → N={min_pass}: {cond_lat:.0f}ms)")
    else:
        logger.info("\n  ❌ 어떤 조건도 품질 기준(PASS)을 만족하지 않음.")
        logger.info("  → CoC가 Action Expert 품질에 필수적임. CUDA Graph 집중 투자 근거 확정.")


# ── 메인 ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="실험 1: Decode Skip / Adaptive Decode sweep")
    p.add_argument("--model",   default="nvidia/Alpamayo-1.5-10B",
                   help="HuggingFace 모델 ID 또는 로컬 경로 (기본: nvidia/Alpamayo-1.5-10B)")
    p.add_argument("--dtype",   default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--runs",    type=int, default=10,
                   help="각 max_generation_length 조건의 반복 실행 횟수 (기본 10)")
    p.add_argument("--tokens",  type=int, nargs="+", default=TOKEN_SWEEP,
                   help=f"테스트할 max_coc_tokens 값 목록 (기본: {TOKEN_SWEEP})")
    p.add_argument("--mock",    action="store_true",
                   help="Mock 모드: 실제 모델 없이 파이프라인 테스트")
    p.add_argument("--output_dir", default="evaluation/results/streaming/exp1_decode_skip",
                   help="결과 저장 디렉토리")
    p.add_argument("--seed",    type=int, default=42,
                   help="입력 텐서 생성용 고정 시드")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  실험 1: Decode Skip / Adaptive Decode")
    logger.info("  목적: CoC 토큰 수와 궤적 품질의 관계 실측")
    logger.info("=" * 65)
    logger.info(f"  sweep 조건: {sorted(set(args.tokens))}")
    logger.info(f"  runs/조건:  {args.runs}")
    logger.info(f"  dtype:      {args.dtype}")
    logger.info(f"  mock 모드:  {args.mock}")
    logger.info(f"  출력 경로:  {out_dir.resolve()}")
    logger.info("=" * 65)

    # ── 모델 로드 + 입력 준비 ──────────────────────────────────────────────────
    if args.mock:
        model = processor = helper = None
        fixed_inp = _make_mock_input()
        logger.info("Mock 모드 — 모델 로딩 생략, mock 입력 사용")
    else:
        model_path = str(Path(args.model).expanduser())
        model, processor, helper = load_model(model_path, args.dtype)

        # 입력 데이터: 기존 프로파일링과 동일한 PhysicalAI 클립 사용
        # → baseline 5,009ms와 직접 비교 가능한 조건 유지
        logger.info("입력 데이터 로드 중 (PhysicalAI 클립)...")
        fixed_inp = load_physicalai_input()

        # 워밍업 (GPU 클럭 안정화, JIT 컴파일 완료)
        # 워밍업도 동일 입력으로 실행해야 본 실험과 동일한 연산 패턴이 캐시된다.
        logger.info("워밍업 2회 실행 중...")
        for _w in range(2):
            _ = run_single_inference(
                model, processor, helper,
                fixed_inp,
                max_coc_tokens=16,
                seed=99,
            )
        logger.info("워밍업 완료 ✅")

    # ── baseline (N=16) 먼저 실행 → 기준 궤적 확보 ───────────────────────────
    token_list = sorted(set(args.tokens))

    # 16이 목록에 없어도 baseline은 반드시 실행
    if 16 not in token_list:
        token_list = sorted(set(token_list + [16]))
        logger.warning("16(baseline)이 --tokens에 없어 자동 추가됨.")

    # baseline을 맨 앞으로 이동
    token_list = [16] + [t for t in token_list if t != 16]

    all_results: list[dict] = []
    baseline_waypoints: Optional[np.ndarray] = None

    for max_gen in token_list:
        cond_result = run_condition(
            max_gen         = max_gen,
            n_runs          = args.runs,
            model           = model,
            processor       = processor,
            helper          = helper,
            fixed_input     = fixed_inp,
            baseline_waypoints = baseline_waypoints,
            use_mock        = args.mock,
            out_dir         = out_dir,
        )
        all_results.append(cond_result)

        # baseline 궤적 저장 (이후 조건들의 ADE/FDE 계산 기준)
        if max_gen == 16:
            baseline_wp_path = out_dir / "baseline_waypoints.npy"
            if baseline_wp_path.exists():
                baseline_waypoints = np.load(str(baseline_wp_path))
            else:
                baseline_waypoints = np.load(str(out_dir / "waypoints_16.npy"))
            np.save(str(baseline_wp_path), baseline_waypoints)
            logger.info(f"기준 궤적 저장 완료: {baseline_wp_path}")

    # baseline을 원래 순서로 돌려서 요약 출력
    all_results_sorted = sorted(all_results, key=lambda r: r["max_coc_tokens"])

    # ── 요약 출력 ──────────────────────────────────────────────────────────────
    print_summary_table(all_results_sorted)

    # ── 전체 요약 JSON 저장 ────────────────────────────────────────────────────
    # 품질 유지 하한 분석
    pass_tokens = [
        r["max_coc_tokens"]
        for r in all_results_sorted
        if r["verdict"] == "PASS"
    ]
    min_pass_token = min(pass_tokens) if pass_tokens else None

    summary = {
        "experiment":          "exp1_decode_skip",
        "date":                time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform":            "Jetson AGX Thor",
        "model_dtype":         args.dtype,
        "mock_mode":           args.mock,
        "n_runs_per_condition": args.runs,
        "input_seed":          args.seed,
        "token_sweep":         token_list,
        "quality_threshold": {
            "ade_pass_m":      ADE_PASS,
            "ade_margin_m":    ADE_MARGIN,
            "fde_pass_m":      FDE_PASS,
            "fde_margin_m":    FDE_MARGIN,
        },
        "min_pass_token_count": min_pass_token,
        "conditions":          all_results_sorted,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\n전체 요약 저장: {summary_path}")

    # ── 최종 결론 출력 ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  실험 1 완료")
    if min_pass_token is not None:
        baseline_rec = next(r for r in all_results_sorted if r["max_coc_tokens"] == 16)
        pass_rec     = next(r for r in all_results_sorted if r["max_coc_tokens"] == min_pass_token)
        bl_lat  = baseline_rec["latency"]["mean_ms"]
        pa_lat  = pass_rec["latency"]["mean_ms"]
        speedup = bl_lat / pa_lat
        logger.info(f"  품질 유지 최소 토큰 수: {min_pass_token}")
        logger.info(f"  새 Decode 하한:        {pass_rec['latency']['mean_ms']:.0f}ms")
        logger.info(f"  baseline 대비 개선:    {speedup:.2f}× "
                    f"({bl_lat:.0f}ms → {pa_lat:.0f}ms)")
        logger.info(f"  → 다음 단계: Exp 7 (EOS Sync 격리) 또는 Exp 6 (CUDA Graph)")
    else:
        logger.info("  품질 기준을 만족하는 조건 없음.")
        logger.info("  → CoC 필요성 실증. Exp 6 (CUDA Graph) + Exp 5 (torch.compile) 집중.")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
