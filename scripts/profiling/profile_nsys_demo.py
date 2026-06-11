"""
profile_nsys_demo.py  ·  v1.0
────────────────────────────────────────────────────────────────────────────────
Alpamayo 1.5  NSIGHT SYSTEMS 전용 데모 프로파일러

목적:
  - Nsight Systems에서 "토크나이징(CPU) → GPU 추론" 흐름을 명확하게 시각화
  - NVTX 마커로 각 단계에 이름 붙이기
  - Python sampling으로 실제 함수 이름 캡처
  - CPU idle / GPU idle 구간을 직접 보여주기

실행 방법 (Thor에서):
  nsys profile \
      --trace=cuda,python,osrt,nvtx \
      --python-sampling=true \
      --python-sampling-frequency=1000 \
      --output=~/alpamayo1.5/profiling_results/nsys_demo \
      python scripts/profiling/profile_nsys_demo.py

결과 파일을 Windows로 복사:
  scp ice401@100.95.177.101:~/alpamayo1.5/profiling_results/nsys_demo.nsys-rep .

Nsight Systems GUI에서 확인:
  1. CPU Thread 레인  → tokenizer.encode(), torch.randn() 등 함수 이름 표시
  2. NVTX 레인        → [TOKENIZE] [VISION] [PREFILL] [DECODE] [FLOW] 컬러 블록
  3. GPU Stream 레인  → [TOKENIZE] 구간 동안 GPU kernel이 없음 (GPU idle 증명)
"""

import sys
import time
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx

# ─── 프로젝트 경로 설정 ───────────────────────────────────────────────────────
# profile_alpamayo.py 와 동일한 경로 설정
ROOT = Path(__file__).resolve().parents[2]   # ~/alpamayo1.5/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# profile_alpamayo.py L795-797 과 동일한 import 경로 사용
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

# ─── NVTX 색상 코드 (Nsight에서 구간별 색상 구분) ────────────────────────────
# 색상은 0xAARRGGBB 형식
COLOR_TOKENIZE = 0xFF_FF6B6B   # 빨간 계열  ← CPU 전용 구간
COLOR_GPU_WAIT = 0xFF_FFD93D   # 노란 계열  ← GPU 대기 구간
COLOR_VISION   = 0xFF_4878CF   # 파랑
COLOR_PREFILL  = 0xFF_6ACC65   # 초록
COLOR_DECODE   = 0xFF_D65F5F   # 빨강
COLOR_FLOW     = 0xFF_B47CC7   # 보라
COLOR_DETOK    = 0xFF_FF6B6B   # 빨간 계열  ← CPU 전용 구간


def nvtx_push(label: str, color: int = 0xFF_AAAAAA):
    """색상 있는 NVTX 마커 시작."""
    nvtx.range_push(label)   # PyTorch는 색상 API 미지원 → 이름으로 구분


def nvtx_pop():
    nvtx.range_pop()


# ─── 메인 데모 ────────────────────────────────────────────────────────────────

def run_nsys_demo(model_path: str, warmup: int = 1, measure: int = 2):
    print("=" * 70)
    print("  Alpamayo 1.5  Nsight Systems Demo Profiler")
    print("  목적: 토크나이징(CPU) vs GPU 추론 구간 시각화")
    print("=" * 70)

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    print("\n[1/4] 모델 로드 중 (이 구간은 nsys 캡처 대상 아님)...")
    torch.cuda.reset_peak_memory_stats()
    model = Alpamayo1_5.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    ).cuda().eval()
    torch.cuda.synchronize()
    print("      모델 로드 완료.")

    # ── 원시 데이터 로드 (토크나이징 이전 상태) ───────────────────────────────
    print("[2/4] 원시 데이터 로드 중...")
    clip_id  = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    data     = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    processor = helper.get_processor(model.tokenizer)

    # 이미지 텐서만 미리 GPU로 이동 (토크나이징과 분리)
    ego_xyz = data["ego_history_xyz"]
    ego_rot = data["ego_history_rot"]
    print("      원시 데이터 로드 완료 (토크나이징은 추론 루프 내에서 실행).")

    # ── 워밍업 ─────────────────────────────────────────────────────────────────
    print(f"\n[3/4] 워밍업 {warmup}회 실행...")
    for w in range(warmup):
        # 워밍업은 NVTX 없이 실행
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data"  : inputs,
            "ego_history_xyz" : ego_xyz,
            "ego_history_rot" : ego_rot,
        }
        model_inputs = helper.to_device(model_inputs, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=0.98, temperature=0.6,
                num_traj_samples=1, return_extra=False,
            )
        torch.cuda.synchronize()
        print(f"      워밍업 {w+1}/{warmup} 완료")

    # ── 실측 (NVTX 마커 포함) ─────────────────────────────────────────────────
    print(f"\n[4/4] 실측 {measure}회 실행 (NVTX 마커 + Python sampling 활성화)...")
    print("      이 구간이 Nsight에서 보여야 하는 내용임.")
    print()

    for run_id in range(measure):
        print(f"  ── Run {run_id + 1} / {measure} ──")

        # ════════════════════════════════════════════════════════════════
        # 구간 A: 토크나이징  (CPU 전용, GPU idle)
        # Nsight에서: NVTX 레인에 [TOKENIZE] 블록 표시
        #             GPU Stream에 kernel 없음 → GPU idle 증명
        # ════════════════════════════════════════════════════════════════
        torch.cuda.synchronize()   # GPU 완료 보장 후 토크나이징 시작
        t_tok_s = time.perf_counter()

        nvtx_push("TOKENIZE  (CPU only — GPU idle here)")
        # ↓ 이 함수가 nsys Python sampling에서 보여야 할 함수들
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,                  # ← tokenizer.encode() 내부 실행
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        nvtx_pop()   # TOKENIZE 끝

        t_tok_ms = (time.perf_counter() - t_tok_s) * 1000
        print(f"    [TOKENIZE]  {t_tok_ms:.1f} ms  (GPU는 이 구간 동안 idle)")

        # 토큰화 결과를 GPU로 이동
        nvtx_push("HOST_TO_DEVICE  (tokenized → GPU)")
        model_inputs = {
            "tokenized_data"  : inputs,
            "ego_history_xyz" : ego_xyz,
            "ego_history_rot" : ego_rot,
        }
        model_inputs = helper.to_device(model_inputs, "cuda")
        torch.cuda.synchronize()
        nvtx_pop()

        # ════════════════════════════════════════════════════════════════
        # 구간 B: GPU 추론  (Vision → Prefill → Decode → Flow)
        # 각 단계는 alpamayo 내부 패치에서 NVTX 마커를 이미 삽입함
        # 여기서는 전체 추론 구간을 하나의 상위 NVTX로 감쌈
        # ════════════════════════════════════════════════════════════════
        t_inf_s = time.perf_counter()
        nvtx_push("GPU_INFERENCE  (Vision + Prefill + Decode + Flow)")

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = \
                model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    return_extra=True,
                )

        torch.cuda.synchronize()
        nvtx_pop()   # GPU_INFERENCE 끝

        t_inf_ms = (time.perf_counter() - t_inf_s) * 1000

        # ════════════════════════════════════════════════════════════════
        # 구간 C: 디토크나이징  (CPU 전용, GPU idle)
        # generated token IDs → 텍스트 변환
        # ════════════════════════════════════════════════════════════════
        t_detok_s = time.perf_counter()
        nvtx_push("DETOKENIZE  (CPU only — GPU idle here)")

        # reasoning trace 토큰 ID 추출
        if extra and "generated_token_ids" in extra:
            generated_ids = extra["generated_token_ids"]
            # token IDs → 텍스트 (CPU에서 실행)
            reasoning_text = model.tokenizer.decode(
                generated_ids[0].cpu().tolist(),
                skip_special_tokens=True,
            )
        else:
            reasoning_text = "(reasoning trace unavailable)"

        nvtx_pop()   # DETOKENIZE 끝
        t_detok_ms = (time.perf_counter() - t_detok_s) * 1000

        # ── Run 요약 출력 ──────────────────────────────────────────────
        total_ms = t_tok_ms + t_inf_ms + t_detok_ms
        print(f"    [GPU_INF ]  {t_inf_ms:.1f} ms")
        print(f"    [DETOK  ]  {t_detok_ms:.1f} ms  (GPU는 이 구간 동안 idle)")
        print(f"    [TOTAL  ]  {total_ms:.1f} ms")
        print(f"    [REASON ]  {reasoning_text[:80]}...")
        print()

    print("=" * 70)
    print("  완료. nsys-rep 파일을 Windows로 복사하세요:")
    print()
    print("  scp ice401@100.95.177.101:\\ ")
    print("      ~/alpamayo1.5/profiling_results/nsys_demo.nsys-rep .")
    print()
    print("  Nsight Systems GUI에서 확인할 사항:")
    print("  1. NVTX row → [TOKENIZE] 블록 위치 확인")
    print("  2. GPU Stream row → TOKENIZE 구간에서 kernel 없음 (빈 공간)")
    print("  3. CPU Thread row → tokenizer.encode() / apply_chat_template 함수명 표시")
    print("  4. [DETOKENIZE] 구간도 동일하게 GPU idle")
    print("=" * 70)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Alpamayo 1.5 Nsight Systems Demo Profiler"
    )
    parser.add_argument(
        "--model-path", type=str,
        default="nvidia/Alpamayo-1.5-10B",
        help="모델 경로 또는 HuggingFace repo ID",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="워밍업 횟수 (NVTX 없이 실행, 기본값: 1)",
    )
    parser.add_argument(
        "--runs", type=int, default=2,
        help="실측 횟수 (NVTX 포함, 기본값: 2)",
    )
    args = parser.parse_args()

    run_nsys_demo(
        model_path=args.model_path,
        warmup=args.warmup,
        measure=args.runs,
    )
