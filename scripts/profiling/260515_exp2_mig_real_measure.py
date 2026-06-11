#!/usr/bin/env python3
"""
EXP-2 Step 2: 실제 MIG 인스턴스에서 측정
────────────────────────────────────────────────────────────────────
실행 전 필수 조건:
  sudo bash ~/alpamayo1.5/scripts/profiling/260515_exp2_mig_setup.sh

이 스크립트가 측정하는 것:

[측정 1] 각 MIG 슬라이스의 compute 성능
  - Vision UUID → 실제 ViT-L 크기 GEMM 실행 → 몇 ms?
  - VLM UUID   → 실제 LM 크기 GEMM 실행   → 몇 ms?
  - Action UUID → 실제 Flow 크기 GEMM 실행 → 몇 ms?
  - Full GPU (MIG 없이) → 동일 GEMM → baseline

[측정 2] MIG 인스턴스 간 텐서 전달 비용
  - Vision UUID → (텐서 생성) → CPU → Action UUID (받아서 GPU에 올림)
  - 실제로 GPU A에서 만든 텐서를 GPU B에서 쓰려면 반드시 CPU 경유

[측정 3] 3-프로세스 파이프라인 E2E 레이턴시
  - Process A (Vision UUID) → Process B (VLM UUID) → Process C (Action UUID)
  - 각 단계 완료 후 텐서 전달 → 실제 단일 프레임 총 시간

실행:
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  python3 ~/alpamayo1.5/scripts/profiling/260515_exp2_mig_real_measure.py

출력:
  profiling_results/260515_exp2/real_mig_results.json
  profiling_results/260515_exp2/real_mig_results.md

작성일: 2026-05-15
"""

import json
import logging
import multiprocessing as mp
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path("profiling_results/260515_exp2")
UUID_FILE = BASE_DIR / "mig_uuids.json"
OUT_DIR   = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPEAT = 5
N_WARMUP = 3

# ─────────────────────────────────────────────────────────────────
# 실제 모델 아키텍처 기반 GEMM 크기
# (실제 Alpamayo 1.5 레이어 구조 근사)
# ─────────────────────────────────────────────────────────────────
MODULE_SPECS = {
    "vision": {
        # ViT-L: d_model=1024, ffn=4096, n_layers=24, seq=196(14×14 patches)
        "d_model": 1024,
        "d_ffn":   4096,
        "n_layers": 24,
        "seq":     196,
        "mode":    "gemm",    # compute-bound
        "description": "ViT-L (1.15 GB)",
    },
    "vlm_prefill": {
        # LM Prefill: d_model=4096, ffn=16384, n_layers=32, seq=64
        "d_model": 4096,
        "d_ffn":   16384,
        "n_layers": 32,
        "seq":     64,
        "mode":    "gemm",
        "description": "LM Prefill (15.17 GB)",
    },
    "vlm_decode": {
        # LM Decode: seq=1 (GEMV), n_tok=20
        "d_model": 4096,
        "d_ffn":   16384,
        "n_layers": 32,
        "n_tok":   20,
        "mode":    "gemv",    # BW-bound
        "description": "LM Decode seq=1 (15.17 GB, BW-bound)",
    },
    "action": {
        # Action Expert: d_model=2048, ffn=8192, n_layers=18, n_euler=10
        "d_model": 2048,
        "d_ffn":   8192,
        "n_layers": 18,
        "n_euler": 10,
        "seq":     64,
        "mode":    "gemm",
        "description": "Action Expert (4.56 GB)",
    },
}

# 인스턴스 간 실제 전달 텐서 크기
TRANSFER_SHAPES = {
    "vision_to_vlm":  (1, 256, 4096),   # image features: ~2 MB
    "vlm_to_action":  (1, 20, 4096),    # hidden state: ~0.16 MB
}


# ─────────────────────────────────────────────────────────────────
# 워커 함수 (별도 프로세스에서 실행)
# ─────────────────────────────────────────────────────────────────

def worker_compute(
    uuid: str,
    module: str,
    result_queue: mp.Queue,
    ready_event: mp.Event,
):
    """
    실제 MIG 인스턴스(uuid)에서 module의 GEMM/GEMV를 실행하고
    실행 시간을 result_queue에 넣음.

    CUDA_VISIBLE_DEVICES=uuid 로 해당 MIG 인스턴스만 보이게 함.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    # torch import를 os.environ 설정 후 진행해야 올바른 device를 잡음
    import torch as _torch

    spec = MODULE_SPECS[module]
    device = _torch.device("cuda:0")

    try:
        if spec["mode"] == "gemm":
            d   = spec["d_model"]
            ffn = spec["d_ffn"]
            seq = spec["seq"]
            A = _torch.randn(seq, d,   device=device, dtype=_torch.float16)
            B = _torch.randn(d,  ffn,  device=device, dtype=_torch.float16)
            C = _torch.randn(ffn, d,   device=device, dtype=_torch.float16)
            n_layers = spec["n_layers"]

            def forward():
                h = A
                for _ in range(n_layers):
                    h = h @ B   # up projection
                    h = h @ C   # down projection (seq×d)

        elif spec["mode"] == "gemv":
            d   = spec["d_model"]
            ffn = spec["d_ffn"]
            n_layers = spec["n_layers"]
            n_tok = spec.get("n_tok", 20)
            W_up   = _torch.randn(ffn, d,   device=device, dtype=_torch.float16)
            W_down = _torch.randn(d,   ffn,  device=device, dtype=_torch.float16)
            x = _torch.randn(d, 1, device=device, dtype=_torch.float16)

            def forward():
                for _ in range(n_tok):
                    h = x
                    for _ in range(n_layers):
                        h_up = W_up @ h
                        h    = W_down @ h_up

        # 웜업
        for _ in range(N_WARMUP):
            forward()
        _torch.cuda.synchronize()

        # 측정
        times = []
        for _ in range(N_REPEAT):
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)
            e0.record()
            forward()
            e1.record()
            _torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1))

        result_queue.put({
            "module": module,
            "uuid": uuid,
            "times_ms": times,
            "mean_ms": statistics.mean(times),
            "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "status": "ok",
        })

    except Exception as e:
        result_queue.put({
            "module": module,
            "uuid": uuid,
            "status": "error",
            "error": str(e),
        })
    finally:
        ready_event.set()


def worker_pipeline(
    uuid: str,
    role: str,               # "vision" | "vlm" | "action"
    recv_queue: mp.Queue,    # 이전 단계에서 텐서 받는 큐
    send_queue: mp.Queue,    # 다음 단계로 텐서 보내는 큐
    result_queue: mp.Queue,  # 타이밍 결과 저장
    start_event: mp.Event,   # 모든 프로세스 준비 완료 신호
    done_event: mp.Event,    # 이 단계 완료 신호
    n_frames: int = 5,
):
    """
    실제 MIG 인스턴스(uuid)에서 파이프라인 워커로 동작.

    Vision: 입력 없음 → 계산 → vision feature 전송
    VLM:    vision feature 수신 → 계산 → hidden state 전송
    Action: hidden state 수신 → 계산 → 완료
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    import torch as _torch

    device = _torch.device("cuda:0")
    times = []

    try:
        start_event.wait()  # 모든 프로세스 준비 완료 대기

        for frame_i in range(n_frames):
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)

            if role == "vision":
                # Vision: 입력 없음, 이미지 feature 생성
                d, ffn, seq = 1024, 4096, 196
                A = _torch.randn(seq, d,   device=device, dtype=_torch.float16)
                B = _torch.randn(d,   ffn, device=device, dtype=_torch.float16)
                C = _torch.randn(ffn, d,   device=device, dtype=_torch.float16)

                e0.record()
                h = A
                for _ in range(24):
                    h = h @ B
                    h = h @ C
                output = _torch.randn(*TRANSFER_SHAPES["vision_to_vlm"],
                                      device=device, dtype=_torch.float16)
                e1.record()
                _torch.cuda.synchronize()

                # CPU로 옮겨서 큐로 전달 (MIG 인스턴스 간 전달 = CPU 경유 필수)
                t_send_start = time.perf_counter()
                send_queue.put(output.cpu())
                t_send_end = time.perf_counter()
                transfer_ms = (t_send_end - t_send_start) * 1000

            elif role == "vlm":
                # VLM: vision feature 수신 → Prefill + Decode
                cpu_tensor = recv_queue.get()
                t_recv_end = time.perf_counter()
                gpu_input = cpu_tensor.to(device)   # CPU → 이 MIG 인스턴스 GPU
                _torch.cuda.synchronize()

                d, ffn = 4096, 16384
                W_up   = _torch.randn(ffn, d, device=device, dtype=_torch.float16)
                W_down = _torch.randn(d,  ffn, device=device, dtype=_torch.float16)
                x      = _torch.randn(d,  1,   device=device, dtype=_torch.float16)

                e0.record()
                # Prefill (seq=64, 32 layers)
                A_seq = _torch.randn(64, d,   device=device, dtype=_torch.float16)
                B_ffn = _torch.randn(d,  ffn, device=device, dtype=_torch.float16)
                C_ffn = _torch.randn(ffn, d,  device=device, dtype=_torch.float16)
                h = A_seq
                for _ in range(32):
                    h = h @ B_ffn
                    h = h @ C_ffn
                # Decode (seq=1, 20 tokens, 32 layers)
                for _ in range(20):
                    hv = x
                    for _ in range(32):
                        up   = W_up   @ hv
                        hv   = W_down @ up
                output = _torch.randn(*TRANSFER_SHAPES["vlm_to_action"],
                                      device=device, dtype=_torch.float16)
                e1.record()
                _torch.cuda.synchronize()

                t_send_start = time.perf_counter()
                send_queue.put(output.cpu())
                t_send_end = time.perf_counter()
                transfer_ms = (t_send_end - t_send_start) * 1000

            elif role == "action":
                # Action: hidden state 수신 → Flow
                cpu_tensor = recv_queue.get()
                gpu_input = cpu_tensor.to(device)
                _torch.cuda.synchronize()

                d, ffn = 2048, 8192
                A = _torch.randn(64,  d,   device=device, dtype=_torch.float16)
                B = _torch.randn(d,   ffn, device=device, dtype=_torch.float16)
                C = _torch.randn(ffn, d,   device=device, dtype=_torch.float16)

                e0.record()
                for _ in range(10):   # n_euler
                    h = A
                    for _ in range(18):  # n_layers
                        h = h @ B
                        h = h @ C
                e1.record()
                _torch.cuda.synchronize()
                transfer_ms = 0.0

            gpu_ms = e0.elapsed_time(e1)
            times.append({
                "frame": frame_i,
                "gpu_ms": round(gpu_ms, 2),
                "transfer_ms": round(transfer_ms, 3) if role != "action" else 0.0,
            })

    except Exception as e:
        result_queue.put({"role": role, "uuid": uuid, "status": "error", "error": str(e)})
        done_event.set()
        return

    result_queue.put({
        "role": role,
        "uuid": uuid,
        "status": "ok",
        "frames": times,
        "mean_gpu_ms": round(statistics.mean(t["gpu_ms"] for t in times), 1),
        "mean_transfer_ms": round(statistics.mean(t["transfer_ms"] for t in times), 3),
    })
    done_event.set()


# ─────────────────────────────────────────────────────────────────
# 측정 1: 각 MIG 슬라이스의 compute 성능
# ─────────────────────────────────────────────────────────────────

def measure_compute_per_slice(uuids: dict) -> dict:
    """각 MIG UUID에서 해당 모듈의 GEMM/GEMV를 실행하고 시간 측정."""
    log.info("\n" + "═" * 60)
    log.info("[측정 1] MIG 슬라이스별 Compute 성능")
    log.info("═" * 60)

    assignments = [
        ("vision",      uuids["vision_uuid"],  "Vision (ViT-L)"),
        ("vlm_prefill", uuids["vlm_uuid"],     "VLM Prefill"),
        ("vlm_decode",  uuids["vlm_uuid"],     "VLM Decode (BW-bound)"),
        ("action",      uuids["action_uuid"],  "Action Expert"),
    ]

    results = {}
    for module, uuid, label in assignments:
        log.info(f"\n  ── {label} @ {uuid[:30]}... ──")

        result_q = mp.Queue()
        ready_ev = mp.Event()
        p = mp.Process(
            target=worker_compute,
            args=(uuid, module, result_q, ready_ev),
        )
        p.start()
        p.join(timeout=120)

        if p.is_alive():
            p.terminate()
            log.error(f"    TIMEOUT: {label}")
            results[module] = {"status": "timeout"}
        else:
            r = result_q.get_nowait() if not result_q.empty() else {"status": "no_result"}
            if r["status"] == "ok":
                log.info(f"    → {r['mean_ms']:.1f} ± {r['std_ms']:.1f} ms")
            else:
                log.error(f"    → ERROR: {r.get('error', '?')}")
            results[module] = r

    return results


# ─────────────────────────────────────────────────────────────────
# 측정 2: Full GPU baseline (MIG 없이)
# ─────────────────────────────────────────────────────────────────

def measure_baseline_fullgpu() -> dict:
    """
    MIG 없을 때(전체 GPU) 동일 GEMM을 실행해서 baseline 확인.
    이 프로세스 자체가 Full GPU 프로세스이므로 직접 실행.
    """
    log.info("\n" + "═" * 60)
    log.info("[측정 2] Full GPU Baseline (비교용)")
    log.info("═" * 60)

    results = {}
    for module, spec in MODULE_SPECS.items():
        log.info(f"\n  ── {spec['description']} ──")

        device = torch.device("cuda:0")

        if spec["mode"] == "gemm":
            d, ffn, seq = spec["d_model"], spec["d_ffn"], spec["seq"]
            A = torch.randn(seq, d,  device=device, dtype=torch.float16)
            B = torch.randn(d,  ffn, device=device, dtype=torch.float16)
            C = torch.randn(ffn, d,  device=device, dtype=torch.float16)
            n_layers = spec["n_layers"]
            def forward():
                h = A
                for _ in range(n_layers):
                    h = h @ B
                    h = h @ C

        elif spec["mode"] == "gemv":
            d, ffn = spec["d_model"], spec["d_ffn"]
            n_layers = spec["n_layers"]
            n_tok = spec.get("n_tok", 20)
            W_up   = torch.randn(ffn, d,  device=device, dtype=torch.float16)
            W_down = torch.randn(d,  ffn, device=device, dtype=torch.float16)
            x = torch.randn(d, 1, device=device, dtype=torch.float16)
            def forward():
                for _ in range(n_tok):
                    h = x
                    for _ in range(n_layers):
                        h = W_up @ h
                        h = W_down @ h

        for _ in range(N_WARMUP):
            forward()
        torch.cuda.synchronize()

        times = []
        for _ in range(N_REPEAT):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            forward()
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1))

        mean_ms = statistics.mean(times)
        std_ms  = statistics.stdev(times) if len(times) > 1 else 0.0
        results[module] = {
            "mean_ms": round(mean_ms, 1),
            "std_ms":  round(std_ms, 1),
            "times_ms": [round(t, 1) for t in times],
        }
        log.info(f"    → {mean_ms:.1f} ± {std_ms:.1f} ms")

    return results


# ─────────────────────────────────────────────────────────────────
# 측정 3: 3-프로세스 파이프라인 E2E
# ─────────────────────────────────────────────────────────────────

def measure_e2e_pipeline(uuids: dict, n_frames: int = 5) -> dict:
    """
    실제 MIG 인스턴스 3개에서 파이프라인 실행.
    Vision(UUID-A) → VLM(UUID-B) → Action(UUID-C)
    텐서 전달은 CPU 경유 (실제 MIG 환경과 동일).
    """
    log.info("\n" + "═" * 60)
    log.info("[측정 3] 3-프로세스 파이프라인 E2E")
    log.info("═" * 60)
    log.info(f"  Vision : {uuids['vision_uuid'][:30]}...")
    log.info(f"  VLM    : {uuids['vlm_uuid'][:30]}...")
    log.info(f"  Action : {uuids['action_uuid'][:30]}...")
    log.info(f"  프레임 수: {n_frames}")

    # 큐 및 이벤트 설정
    q_vis_to_vlm  = mp.Queue()
    q_vlm_to_act  = mp.Queue()
    result_q      = mp.Queue()
    start_ev      = mp.Event()
    done_vision   = mp.Event()
    done_vlm      = mp.Event()
    done_action   = mp.Event()

    processes = [
        mp.Process(target=worker_pipeline, args=(
            uuids["vision_uuid"], "vision",
            None, q_vis_to_vlm, result_q, start_ev, done_vision, n_frames,
        )),
        mp.Process(target=worker_pipeline, args=(
            uuids["vlm_uuid"], "vlm",
            q_vis_to_vlm, q_vlm_to_act, result_q, start_ev, done_vlm, n_frames,
        )),
        mp.Process(target=worker_pipeline, args=(
            uuids["action_uuid"], "action",
            q_vlm_to_act, None, result_q, start_ev, done_action, n_frames,
        )),
    ]

    wall_start = time.perf_counter()
    for p in processes:
        p.start()

    # 모든 프로세스 시작 신호
    time.sleep(0.5)   # 프로세스 초기화 대기
    start_ev.set()

    # 완료 대기
    for ev in [done_vision, done_vlm, done_action]:
        ev.wait(timeout=300)

    wall_total_ms = (time.perf_counter() - wall_start) * 1000

    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    # 결과 수집
    results = []
    while not result_q.empty():
        results.append(result_q.get_nowait())

    log.info(f"\n  Wall-clock 총 시간: {wall_total_ms:.0f} ms ({n_frames} 프레임)")
    log.info(f"  프레임당 평균: {wall_total_ms/n_frames:.0f} ms")

    for r in results:
        if r["status"] == "ok":
            log.info(
                f"  {r['role']:8s}: GPU={r['mean_gpu_ms']:.0f}ms  "
                f"Transfer={r['mean_transfer_ms']:.2f}ms"
            )
        else:
            log.error(f"  {r['role']:8s}: ERROR - {r.get('error', '?')}")

    return {
        "wall_total_ms": round(wall_total_ms, 1),
        "wall_per_frame_ms": round(wall_total_ms / n_frames, 1),
        "n_frames": n_frames,
        "worker_results": results,
    }


# ─────────────────────────────────────────────────────────────────
# 보고서
# ─────────────────────────────────────────────────────────────────

def make_report(
    uuids: dict,
    baseline: dict,
    mig_compute: dict,
    e2e: dict,
) -> str:
    lines = [
        "# EXP-2: 실제 MIG 인스턴스 측정 결과",
        "",
        "**실험 일시**: " + time.strftime("%Y-%m-%d %H:%M"),
        "",
        "## MIG 인스턴스 배정",
        "",
        f"| 모듈 | MIG UUID |",
        f"|---|---|",
        f"| Vision (ViT-L, 1.15 GB) | `{uuids['vision_uuid']}` |",
        f"| VLM (15.17 GB) | `{uuids['vlm_uuid']}` |",
        f"| Action Expert (4.56 GB) | `{uuids['action_uuid']}` |",
        "",
        "---",
        "",
        "## [측정 1] MIG 슬라이스 vs Full GPU — Compute 성능",
        "",
        "| 모듈 | Full GPU (ms) | MIG 슬라이스 (ms) | 슬로우다운 |",
        "|---|---|---|---|",
    ]

    module_labels = {
        "vision":      "Vision (ViT-L)",
        "vlm_prefill": "VLM Prefill",
        "vlm_decode":  "VLM Decode",
        "action":      "Action Expert",
    }

    for module, label in module_labels.items():
        b = baseline.get(module, {})
        m = mig_compute.get(module, {})
        b_ms = b.get("mean_ms", "N/A")
        m_ms = m.get("mean_ms", "N/A")
        if isinstance(b_ms, float) and isinstance(m_ms, float):
            slowdown = f"{m_ms/b_ms:.2f}×"
        else:
            slowdown = "N/A"
        lines.append(f"| {label} | {b_ms} | {m_ms} | {slowdown} |")

    lines += [
        "",
        "> MIG 슬라이스 = 전체 SM의 일부만 할당",
        "> compute-bound 모듈은 SM 비례 슬로우다운 예상",
        "> BW-bound 모듈(Decode)은 메모리 BW 비례 슬로우다운 예상",
        "",
        "---",
        "",
        "## [측정 2] 인스턴스 간 텐서 전달 비용",
        "",
        "전달 경로: `GPU(MIG-A) → .cpu() → queue → .to(cuda:0, MIG-B)`",
        "",
    ]

    for r in e2e.get("worker_results", []):
        if r.get("status") == "ok" and r.get("mean_transfer_ms", 0) > 0:
            role = r["role"]
            dest = "VLM" if role == "vision" else "Action"
            shape = TRANSFER_SHAPES.get(
                "vision_to_vlm" if role == "vision" else "vlm_to_action", "?"
            )
            size_mb = 1
            if shape != "?":
                n = 1
                for s in shape:
                    n *= s
                size_mb = n * 2 / 1e6   # float16
            lines.append(
                f"| {role} → {dest} | {shape} | {size_mb:.2f} MB | "
                f"{r['mean_transfer_ms']:.3f} ms |"
            )

    lines += [
        "",
        "---",
        "",
        "## [측정 3] E2E 파이프라인 레이턴시",
        "",
        f"| 구성 | E2E (ms/frame) |",
        f"|---|---|",
        f"| Full GPU 순차 (baseline) | "
        f"{sum(baseline.get(m, {}).get('mean_ms', 0) for m in MODULE_SPECS):.0f} |",
        f"| MIG 3-인스턴스 파이프라인 | **{e2e.get('wall_per_frame_ms', 'N/A')}** |",
    ]

    if e2e.get("wall_per_frame_ms") and e2e["wall_per_frame_ms"] > 0:
        baseline_total = sum(
            baseline.get(m, {}).get("mean_ms", 0) for m in MODULE_SPECS
        )
        slowdown = e2e["wall_per_frame_ms"] / baseline_total if baseline_total else 0
        lines.append(f"| Slowdown | {slowdown:.2f}× |")

    lines += [
        "",
        "---",
        "",
        "## 결론",
        "",
        "### 왜 MIG 배정이 단일 프레임 레이턴시를 줄이지 못하는가",
        "",
        "1. **각 모듈이 더 작은 자원에서 실행** → 측정 1에서 실증",
        "2. **인스턴스 간 전달 비용 추가** → 측정 2에서 정량화",
        "3. **데이터 의존성 유지** → 여전히 순차 실행, 병렬화 불가",
        "",
        "→ MIG의 올바른 용도: **다중 요청 동시 처리 (multi-tenancy)**",
        "→ 단일 프레임 레이턴시 최적화: **크로스프레임 파이프라인 (EXP-3)**",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

def main():
    mp.set_start_method("spawn", force=True)

    log.info("=" * 60)
    log.info("EXP-2: 실제 MIG 인스턴스 측정")
    log.info("=" * 60)

    # UUID 파일 읽기
    if not UUID_FILE.exists():
        log.error(f"UUID 파일 없음: {UUID_FILE}")
        log.error("먼저 실행: sudo bash .../260515_exp2_mig_setup.sh")
        return

    uuids = json.loads(UUID_FILE.read_text())
    log.info(f"Vision UUID : {uuids['vision_uuid']}")
    log.info(f"VLM UUID    : {uuids['vlm_uuid']}")
    log.info(f"Action UUID : {uuids['action_uuid']}")

    # [측정 1] MIG 슬라이스 Compute
    mig_compute = measure_compute_per_slice(uuids)

    # [측정 2 & 3] Full GPU Baseline
    baseline = measure_baseline_fullgpu()

    # [측정 3] E2E 파이프라인
    e2e = measure_e2e_pipeline(uuids, n_frames=5)

    # 저장
    output = {
        "experiment": "EXP-2",
        "uuids": uuids,
        "baseline_full_gpu": baseline,
        "mig_compute": mig_compute,
        "e2e_pipeline": e2e,
    }
    json_path = OUT_DIR / "real_mig_results.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    log.info(f"\n✅ JSON: {json_path}")

    md = make_report(uuids, baseline, mig_compute, e2e)
    md_path = OUT_DIR / "real_mig_results.md"
    md_path.write_text(md, encoding="utf-8")
    log.info(f"✅ MD:   {md_path}")

    log.info("\n실험 완료.")


if __name__ == "__main__":
    main()
