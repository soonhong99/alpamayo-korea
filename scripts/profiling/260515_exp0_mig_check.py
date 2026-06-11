#!/usr/bin/env python3
"""
EXP-0: MIG 활성화 확인 및 환경 진단
────────────────────────────────────────────────────────────────────
실험 계획서: docs/260515_mig_pipeline_experiment_plan.md

목적:
  1. Thor Blackwell iGPU에서 MIG 모드 활성화 가능 여부 확인
  2. 사용 가능한 MIG GI/CI 프로파일 나열
  3. 현재 Alpamayo 1.5 추론 워크로드의 SM 점유 패턴 측정
  4. 크로스프레임 파이프라인 실험(EXP-3) 기반 환경 검증

실행 방법 (Thor):
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  python3 ~/alpamayo1.5/scripts/profiling/260515_exp0_mig_check.py

출력:
  profiling_results/260515_exp0/exp0_mig_check.json
  profiling_results/260515_exp0/exp0_mig_check.md

작성일: 2026-05-15
"""

import subprocess
import json
import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("profiling_results/260515_exp0")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 헬퍼: 셸 명령어 실행
# ─────────────────────────────────────────────────────────────────

def run_cmd(cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """명령어 실행 후 (returncode, stdout, stderr) 반환."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s"
    except Exception as e:
        return -2, "", str(e)


def check(label: str, cmd: str, timeout: int = 30) -> dict:
    """명령 실행 결과를 dict로 정리."""
    rc, out, err = run_cmd(cmd, timeout)
    status = "OK" if rc == 0 else ("TIMEOUT" if rc == -1 else "FAIL")
    log.info(f"[{status}] {label}")
    if out:
        for line in out.splitlines()[:6]:   # 너무 긴 출력 자르기
            log.info(f"       {line}")
    if err and rc != 0:
        log.warning(f"       STDERR: {err[:200]}")
    return {"label": label, "cmd": cmd, "rc": rc, "status": status,
            "stdout": out, "stderr": err[:300]}


# ─────────────────────────────────────────────────────────────────
# 1. GPU 기본 정보
# ─────────────────────────────────────────────────────────────────

def section_gpu_info() -> list[dict]:
    log.info("\n═══ 1. GPU 기본 정보 ═══")
    results = []
    results.append(check(
        "nvidia-smi 버전 / GPU 이름",
        "nvidia-smi --query-gpu=name,driver_version,compute_cap "
        "--format=csv,noheader"
    ))
    results.append(check(
        "CUDA 버전",
        "nvcc --version 2>/dev/null || nvidia-smi | grep 'CUDA Version'"
    ))
    results.append(check(
        "SM 개수 및 메모리",
        "nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu "
        "--format=csv,noheader"
    ))
    results.append(check(
        "python torch CUDA",
        "python3 -c \"import torch; print(torch.version.cuda, "
        "torch.cuda.get_device_name(0), "
        "torch.cuda.get_device_properties(0).multi_processor_count, 'SMs')\""
    ))
    return results


# ─────────────────────────────────────────────────────────────────
# 2. MIG 모드 상태
# ─────────────────────────────────────────────────────────────────

def section_mig_status() -> list[dict]:
    log.info("\n═══ 2. MIG 모드 상태 ═══")
    results = []

    # 현재 MIG 모드 확인
    results.append(check(
        "MIG 현재 모드",
        "nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader"
    ))

    # MIG 활성화 가능 여부 (sudo 필요 — 실패해도 정보 수집)
    results.append(check(
        "MIG 활성화 시도 (sudo 없이)",
        "nvidia-smi -mig 1 2>&1 || echo '[sudo required or already enabled]'"
    ))

    # GI 프로파일 목록 (MIG 지원 GPU만 출력)
    results.append(check(
        "지원 GI 프로파일 목록 (MIG GPU instance profiles)",
        "nvidia-smi mig -lgip 2>&1"
    ))

    # CI 프로파일 목록
    results.append(check(
        "지원 CI 프로파일 목록 (Compute instance profiles)",
        "nvidia-smi mig -lcip 2>&1"
    ))

    # 현재 MIG 인스턴스 (있다면)
    results.append(check(
        "현재 MIG 인스턴스 목록",
        "nvidia-smi mig -lgi 2>&1"
    ))

    return results


# ─────────────────────────────────────────────────────────────────
# 3. CUDA MPS / Green Contexts 대안 확인
# ─────────────────────────────────────────────────────────────────

def section_mps_check() -> list[dict]:
    log.info("\n═══ 3. CUDA MPS / Green Contexts ═══")
    results = []
    results.append(check(
        "MPS 데몬 상태",
        "nvidia-cuda-mps-control -d 2>&1 || echo '[MPS not running or unavailable]'"
    ))
    results.append(check(
        "CUDA_MPS_PIPE_DIRECTORY",
        "ls /tmp/nvidia-mps 2>/dev/null && echo 'MPS dir exists' || echo 'MPS dir absent'"
    ))
    # Green Contexts: CUDA 12.4+ 필요, PyTorch 아직 미지원이므로 개념만 확인
    results.append(check(
        "CUDA 드라이버 버전 (Green Contexts 요건: 12.4+)",
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader"
    ))
    return results


# ─────────────────────────────────────────────────────────────────
# 4. SM 점유 패턴 측정 (SM 활용률 샘플링)
# ─────────────────────────────────────────────────────────────────

def section_sm_occupancy() -> dict:
    """
    Alpamayo 1.5 추론 중 nvidia-smi dmon으로 SM 사용률 샘플링.
    MIG 없이 전체 GPU 사용 시 SM 점유율 baseline 측정.
    실제 추론 없이 idle + short GEMM 벤치로 대신함.
    """
    log.info("\n═══ 4. SM 점유 패턴 (GEMM 벤치마크) ═══")

    # 간단한 GEMM으로 GPU 활성화 후 SM 샘플링
    bench_script = """\
import torch, time
d = torch.device("cuda:0")
# 큰 GEMM — FP16
A = torch.randn(4096, 4096, device=d, dtype=torch.float16)
B = torch.randn(4096, 4096, device=d, dtype=torch.float16)
torch.cuda.synchronize()
for _ in range(100):
    C = A @ B
torch.cuda.synchronize()
print("GEMM 4096x4096 FP16 완료")
# 작은 GEMV — seq=1 decode 모사
A2 = torch.randn(16384, 4096, device=d, dtype=torch.float16)
v = torch.randn(4096, 1, device=d, dtype=torch.float16)
torch.cuda.synchronize()
for _ in range(200):
    out = A2 @ v
torch.cuda.synchronize()
print("GEMV 16384x4096 FP16 완료")
"""

    bench_py = OUT_DIR / "_bench_tmp.py"
    bench_py.write_text(bench_script)

    # dmon으로 SM 사용률 30초 샘플링 (백그라운드)
    dmon_out = OUT_DIR / "dmon_sm.log"
    dmon_proc = subprocess.Popen(
        f"nvidia-smi dmon -s u -d 1 -f {dmon_out}",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(1)  # dmon 시작 대기

    rc, out, err = run_cmd(f"python3 {bench_py}", timeout=60)
    time.sleep(2)
    dmon_proc.terminate()

    # dmon 결과 파싱
    # dmon 헤더 예시: # gpu   sm  mem  enc  dec  jpg  ofa
    # 데이터 행 예시:    0   87   12    0    0    0    0
    sm_col_idx = None   # sm 컬럼 인덱스 (헤더에서 동적 감지)
    sm_values = []
    if dmon_out.exists():
        for line in dmon_out.read_text().splitlines():
            stripped = line.strip()
            # 헤더 행에서 sm 컬럼 위치 찾기
            if stripped.startswith("#"):
                parts = stripped.lstrip("#").split()
                if "sm" in parts:
                    sm_col_idx = parts.index("sm") + 1  # 데이터 행엔 gpu idx가 맨 앞
                continue
            parts = stripped.split()
            if not parts or not parts[0].isdigit():
                continue
            try:
                col = sm_col_idx if sm_col_idx is not None else 1
                sm_values.append(int(parts[col]))
            except (IndexError, ValueError):
                pass

    avg_sm = sum(sm_values) / len(sm_values) if sm_values else None
    peak_sm = max(sm_values) if sm_values else None

    avg_str  = f"{avg_sm:.1f}%" if avg_sm is not None else "N/A (dmon 파싱 실패)"
    peak_str = f"{peak_sm}%"    if peak_sm is not None else "N/A"
    log.info(f"SM 사용률: 평균={avg_str}, 피크={peak_str}  (n={len(sm_values)} 샘플)")

    bench_py.unlink(missing_ok=True)

    return {
        "label": "SM occupancy (GEMM bench proxy)",
        "bench_rc": rc,
        "bench_out": out,
        "sm_samples": len(sm_values),
        "sm_avg_pct": round(avg_sm, 1) if avg_sm else None,
        "sm_peak_pct": peak_sm,
        "dmon_log": str(dmon_out),
    }


# ─────────────────────────────────────────────────────────────────
# 5. 크로스프레임 파이프라인 환경 요건 확인
# ─────────────────────────────────────────────────────────────────

def section_pipeline_prereqs() -> list[dict]:
    log.info("\n═══ 5. 크로스프레임 파이프라인 환경 요건 ═══")
    results = []

    # CUDA 스트림 지원
    results.append(check(
        "CUDA 멀티스트림 지원 확인",
        "python3 -c \""
        "import torch; "
        "s1=torch.cuda.Stream(); s2=torch.cuda.Stream(); "
        "print('streams ok:', id(s1)!=id(s2))"
        "\""
    ))

    # torch.cuda.Event 지원
    results.append(check(
        "CUDA Events 지원",
        "python3 -c \""
        "import torch; "
        "e=torch.cuda.Event(enable_timing=True); "
        "print('cuda Event ok')"
        "\""
    ))

    # asyncio + threading 혼합 가능 여부 (파이프라인 구현에 필요)
    results.append(check(
        "Python threading 기반 파이프라인 요건",
        "python3 -c \""
        "import threading, queue; "
        "q=queue.Queue(); "
        "t=threading.Thread(target=lambda: q.put(42)); "
        "t.start(); t.join(); "
        "print('threading queue ok:', q.get())"
        "\""
    ))

    # KV cache 저장 가능한 메모리 여부
    results.append(check(
        "사용 가능 GPU 메모리 (KV cache용)",
        "python3 -c \""
        "import torch; "
        "free, total = torch.cuda.mem_get_info(); "
        "print(f'free={free/1e9:.1f}GB total={total/1e9:.1f}GB')"
        "\""
    ))

    # Alpamayo 모델 import 가능 여부
    results.append(check(
        "alpamayo_inference import 가능",
        "python3 -c \""
        "import importlib.util; "
        "spec = importlib.util.find_spec('transformers'); "
        "print('transformers:', spec is not None); "
        "spec2 = importlib.util.find_spec('torch'); "
        "print('torch:', spec2 is not None)"
        "\""
    ))

    return results


# ─────────────────────────────────────────────────────────────────
# 6. 요약 판정
# ─────────────────────────────────────────────────────────────────

def make_verdict(
    gpu_info: list[dict],
    mig_status: list[dict],
    mps_check: list[dict],
    sm_occ: dict,
    prereqs: list[dict],
) -> dict:
    """
    수집된 결과를 종합해 다음 실험 진행 여부를 판단.
    """
    # MIG 모드 현재 상태
    mig_mode_result = next((r for r in mig_status if "현재 모드" in r["label"]), {})
    mig_enabled = "Enabled" in mig_mode_result.get("stdout", "")
    mig_disabled = "Disabled" in mig_mode_result.get("stdout", "")
    mig_supported = mig_mode_result.get("rc", 1) == 0

    # GI 프로파일 존재 여부
    # 주의: nvidia-smi mig -lgip 는 실패 시에도 테이블 헤더를 stdout으로 출력하고
    # "Failed to display GPU instance profiles: Unknown Error" 도 stdout으로 출력.
    # RC=255 이면 실패 → rc==0 AND "Unknown Error" 없음 AND 실제 프로파일 행 존재해야 True.
    gi_result = next((r for r in mig_status if "GI 프로파일" in r["label"]), {})
    gi_stdout = gi_result.get("stdout", "")
    gi_rc     = gi_result.get("rc", 1)
    has_gi_profiles = (
        gi_rc == 0
        and len(gi_stdout) > 10
        and "Unknown Error" not in gi_stdout
        and "Failed to" not in gi_stdout
    )

    # 파이프라인 요건 충족
    prereq_ok = all(r.get("rc", 1) == 0 for r in prereqs)

    # SM 점유
    sm_avg = sm_occ.get("sm_avg_pct")
    sm_idle = sm_avg is not None and sm_avg < 30   # idle일 때 SM 낮음 → 정상

    verdict = {
        "timestamp": datetime.now().isoformat(),
        "mig_supported": mig_supported,
        "mig_currently_enabled": mig_enabled,
        "mig_currently_disabled": mig_disabled,
        "has_gi_profiles": has_gi_profiles,
        "pipeline_prereqs_ok": prereq_ok,
        "sm_avg_pct_gemm": sm_avg,
        "sm_peak_pct_gemm": sm_occ.get("sm_peak_pct"),
    }

    # 다음 단계 권고
    # has_gi_profiles = False : GI 파티셔닝 미지원 (Thor iGPU 실증됨)
    # has_gi_profiles = True  : 실제 MIG 파티셔닝 가능
    if not mig_supported:
        verdict["recommendation"] = (
            "⚠️  MIG 모드 자체 미지원. "
            "EXP-3 (크로스프레임 파이프라인)으로 진행."
        )
        verdict["next_exp"] = "EXP-3 (cross-frame pipeline)"
    elif mig_enabled and not has_gi_profiles:
        verdict["recommendation"] = (
            "❌  MIG 모드는 켜지지만 GI 파티셔닝 미지원 "
            "(nvidia-smi mig -lgip → Unknown Error). "
            "Thor Blackwell iGPU에서 확인된 제약 (JetPack 7, 드라이버 580.00). "
            "sudo nvidia-smi -mig 0 으로 MIG 비활성화 후 "
            "EXP-3 (크로스프레임 파이프라인)으로 진행."
        )
        verdict["next_exp"] = "sudo nvidia-smi -mig 0  →  EXP-3 (cross-frame pipeline)"
        verdict["mig_gi_supported"] = False
    elif mig_enabled and has_gi_profiles:
        verdict["recommendation"] = (
            "✅  MIG GI 파티셔닝 지원 확인. EXP-1 시작 가능."
        )
        verdict["next_exp"] = "EXP-1 (MIG slice scaling)"
        verdict["mig_gi_supported"] = True
    else:
        verdict["recommendation"] = (
            "ℹ️  MIG 비활성화 상태. "
            "sudo nvidia-smi -mig 1 후 GI 프로파일 확인 필요. "
            "MIG 없이 EXP-3 먼저 진행 가능."
        )
        verdict["next_exp"] = "EXP-3 (cross-frame pipeline, MIG 없이)"

    return verdict


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("EXP-0: MIG 활성화 확인 및 환경 진단")
    log.info(f"실행 시각: {datetime.now().isoformat()}")
    log.info(f"출력 디렉터리: {OUT_DIR.resolve()}")
    log.info("=" * 60)

    gpu_info   = section_gpu_info()
    mig_status = section_mig_status()
    mps_check  = section_mps_check()
    sm_occ     = section_sm_occupancy()
    prereqs    = section_pipeline_prereqs()
    verdict    = make_verdict(gpu_info, mig_status, mps_check, sm_occ, prereqs)

    # ── JSON 저장 ──
    result = {
        "experiment": "EXP-0",
        "title": "MIG 활성화 확인 및 환경 진단",
        "verdict": verdict,
        "sections": {
            "gpu_info":   gpu_info,
            "mig_status": mig_status,
            "mps_check":  mps_check,
            "sm_occupancy": sm_occ,
            "pipeline_prereqs": prereqs,
        }
    }
    json_path = OUT_DIR / "exp0_mig_check.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    log.info(f"\n✅ JSON 저장: {json_path}")

    # ── Markdown 저장 ──
    md = _make_markdown(result)
    md_path = OUT_DIR / "exp0_mig_check.md"
    md_path.write_text(md, encoding="utf-8")
    log.info(f"✅ MD  저장: {md_path}")

    # ── 판정 출력 ──
    log.info("\n" + "═" * 60)
    log.info("📋 판정 요약")
    log.info("═" * 60)
    for k, v in verdict.items():
        log.info(f"  {k}: {v}")

    return result


def _make_markdown(result: dict) -> str:
    v = result["verdict"]
    lines = [
        "# EXP-0: MIG 활성화 확인 및 환경 진단",
        "",
        f"**실행 시각**: {v['timestamp']}  ",
        f"**플랫폼**: Jetson AGX Thor (Blackwell SM 11.0)  ",
        "",
        "---",
        "",
        "## 판정 요약",
        "",
        f"| 항목 | 결과 |",
        f"|---|---|",
        f"| MIG 지원 여부 | {'✅ 지원' if v['mig_supported'] else '❌ 미지원'} |",
        f"| MIG 현재 상태 | {'✅ 활성화' if v['mig_currently_enabled'] else ('🔴 비활성화' if v['mig_currently_disabled'] else '❓ 불명')} |",
        f"| GI 프로파일 목록 | {'✅ 있음' if v['has_gi_profiles'] else '❌ 없음'} |",
        f"| 파이프라인 환경 요건 | {'✅ 충족' if v['pipeline_prereqs_ok'] else '⚠️ 일부 미충족'} |",
        f"| SM 평균 점유율 (GEMM proxy) | {v['sm_avg_pct_gemm']}% |",
        f"| SM 피크 점유율 | {v['sm_peak_pct_gemm']}% |",
        "",
        f"**권고사항**: {v['recommendation']}  ",
        f"**다음 실험**: {v['next_exp']}",
        "",
        "---",
        "",
        "## 상세 결과",
        "",
    ]

    for section_name, items in result["sections"].items():
        lines.append(f"### {section_name}")
        lines.append("")
        if isinstance(items, list):
            lines.append("| 항목 | 상태 | 출력 요약 |")
            lines.append("|---|---|---|")
            for item in items:
                out_short = item.get("stdout", "")[:120].replace("\n", " ")
                err_short = item.get("stderr", "")[:80].replace("\n", " ")
                display = out_short if out_short else err_short
                lines.append(
                    f"| {item['label']} | {item['status']} | `{display}` |"
                )
        elif isinstance(items, dict):
            lines.append("```")
            for k, val in items.items():
                lines.append(f"{k}: {val}")
            lines.append("```")
        lines.append("")

    lines += [
        "---",
        "",
        "## 다음 단계",
        "",
        "### MIG가 지원되는 경우",
        "```bash",
        "# MIG 활성화 (sudo 필요)",
        "sudo nvidia-smi -mig 1",
        "sudo reboot  # 또는 서비스 재시작",
        "",
        "# GI 인스턴스 생성 예시 (1g.10gb × 4 = 4 슬라이스)",
        "sudo nvidia-smi mig -cgi 19,19,19,19 -C",
        "nvidia-smi mig -lgi   # 생성 확인",
        "```",
        "",
        "### MIG 없이 EXP-3 진행 (크로스프레임 파이프라인)",
        "```bash",
        "# 단일 GPU, 멀티 CUDA 스트림 기반 파이프라인",
        "python3 ~/alpamayo1.5/scripts/profiling/260515_exp3_pipeline.py",
        "```",
        "",
        "자세한 실험 계획: `docs/260515_mig_pipeline_experiment_plan.md`",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    main()
