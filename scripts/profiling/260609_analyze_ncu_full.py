"""
260609_analyze_ncu_full.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
260609_run_ncu_full.sh 실행 후 CSV 파싱 → DRAM 대역폭 분석 출력

파싱 대상 (results_dir/):
  ncu_ve.csv          ← VE
  ncu_prefill.csv     ← LM Prefill
  ncu_decode_all.csv  ← Decode 전체 (EOS까지) ★ 수정판
  ncu_flow.csv        ← Flow ODE
  timing_results.json ← 타이밍 기준값

SM 11.0 metrics 변환:
  lts__d_sectors_fill_sysmem.sum              × 32 / 1e9 = DRAM read GB
  lts__t_sectors_aperture_sysmem_op_write.sum × 32 / 1e9 = DRAM write GB
  lts__t_request_hit_rate.pct                            = L2 hit rate

CSV 파싱 방식:
  ncu CSV는 NVTX 컬럼에 중첩 큰따옴표가 많아서 csv.reader가 실패함.
  → line.split('"') 방식 사용:
    parts[-6] = metric name
    parts[-2] = value (콤마 포함 수치, "1,234" → "1234" 변환 후 float)

사용법:
  python3 260609_analyze_ncu_full.py
  python3 260609_analyze_ncu_full.py --results-dir /path/to/results
  python3 260609_analyze_ncu_full.py --results-dir ... --timing-json ...
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════════════

DRAM_PEAK_GBS  = 231.0       # Thor LPDDR5X 피크
SECTOR_BYTES   = 32          # LTS sector = 32 bytes (SM 11.0)
FLOW_ODE_STEPS = 10          # 실측 확정값

# SM 11.0 metric 이름
M_DRAM_READ  = "lts__d_sectors_fill_sysmem.sum"
M_DRAM_WRITE = "lts__t_sectors_aperture_sysmem_op_write.sum"
M_L2_HIT     = "lts__t_request_hit_rate.pct"


# ═══════════════════════════════════════════════════════════════════════
# CSV 파싱
# ═══════════════════════════════════════════════════════════════════════

def parse_ncu_csv(csv_path: Path) -> dict:
    """
    ncu --csv 출력 파싱.

    반환:
      {
        "n_kernels": int,
        "n_metrics": int,
        "dram_read_sectors": float,   # sum of M_DRAM_READ
        "dram_write_sectors": float,  # sum of M_DRAM_WRITE
        "l2_hit_rates": list[float],  # per-kernel hit rates
        "dram_read_GB": float,        # sectors × 32 / 1e9
        "dram_write_GB": float,
        "dram_total_GB": float,
        "l2_hit_pct": float,          # mean
      }
    """
    if not csv_path.exists():
        log.warning(f"  파일 없음: {csv_path}")
        return {}

    read_sectors  = 0.0
    write_sectors = 0.0
    l2_hit_rates  = []
    n_kernels     = 0    # Measure 행 커널 수 추정 (metric/kernel 당 1행)
    n_metrics     = 0    # 파싱된 metric 행 수

    metric_set = set()   # 발견된 metric 이름들

    # target metric 이름을 미리 집합으로 준비 (빠른 in-check)
    _target_metrics = (M_DRAM_READ, M_DRAM_WRITE, M_L2_HIT)

    with open(csv_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            # ★ 수정 (260609): ncu CSV 행은 숫자 ID("1","2"...)로 시작함
            #   startswith('"Measure"') 는 한 행도 매칭 못 함 → 0 커널
            #   → target metric 이름이 포함된 행만 파싱 (빠르고 정확)
            if not any(m in line for m in _target_metrics):
                continue

            # line.split('"') 방식 파싱
            parts = line.split('"')
            if len(parts) < 6:
                continue

            try:
                metric_name = parts[-6].strip(' ,')
                val_raw     = parts[-2].strip()
                value       = float(val_raw.replace(",", ""))
            except (IndexError, ValueError):
                continue

            metric_set.add(metric_name)
            n_metrics += 1

            if metric_name == M_DRAM_READ:
                read_sectors  += value
                n_kernels     += 1  # 각 커널당 read metric 1개
            elif metric_name == M_DRAM_WRITE:
                write_sectors += value
            elif metric_name == M_L2_HIT:
                l2_hit_rates.append(value)

    # sectors → bytes → GB
    read_gb  = read_sectors  * SECTOR_BYTES / 1e9
    write_gb = write_sectors * SECTOR_BYTES / 1e9
    total_gb = read_gb + write_gb
    l2_mean  = sum(l2_hit_rates) / len(l2_hit_rates) if l2_hit_rates else 0.0

    # 커널 수: read + write + l2 = 3 metrics/kernel → n_metrics/3
    est_kernels = n_metrics // len(metric_set) if metric_set else n_kernels

    log.info(f"  {csv_path.name}: {est_kernels} 커널, {n_metrics} metric행 파싱")
    log.info(f"    DRAM read:   {read_gb:.3f} GB  ({read_sectors:,.0f} sectors)")
    log.info(f"    DRAM write:  {write_gb:.3f} GB  ({write_sectors:,.0f} sectors)")
    log.info(f"    DRAM total:  {total_gb:.3f} GB")
    log.info(f"    L2 hit rate: {l2_mean:.1f}%  (mean, {len(l2_hit_rates)} 샘플)")

    return {
        "n_kernels"         : est_kernels,
        "n_metrics"         : n_metrics,
        "dram_read_sectors" : read_sectors,
        "dram_write_sectors": write_sectors,
        "l2_hit_rates"      : l2_hit_rates,
        "dram_read_GB"      : round(read_gb,  3),
        "dram_write_GB"     : round(write_gb, 3),
        "dram_total_GB"     : round(total_gb, 3),
        "l2_hit_pct"        : round(l2_mean,  2),
    }


# ═══════════════════════════════════════════════════════════════════════
# 이론값 계산 보조
# ═══════════════════════════════════════════════════════════════════════

def bw_gbs(dram_gb: float, ms: float) -> float:
    """GB/ms → GB/s"""
    if ms <= 0 or dram_gb <= 0:
        return 0.0
    return dram_gb / (ms / 1000.0)


def overhead_ratio(actual_gb: float, theory_gb: float) -> str:
    if theory_gb <= 0 or actual_gb <= 0:
        return "n/a"
    return f"{actual_gb/theory_gb:.1f}×"


# ═══════════════════════════════════════════════════════════════════════
# 메인 분석
# ═══════════════════════════════════════════════════════════════════════

def analyze(results_dir: Path, timing_json: Path):
    log.info("══════════════════════════════════════════════")
    log.info("CSV 파싱 시작")
    log.info("══════════════════════════════════════════════")

    # CSV 파싱
    ve_data      = parse_ncu_csv(results_dir / "ncu_ve.csv")
    prefill_data = parse_ncu_csv(results_dir / "ncu_prefill.csv")
    decode_data  = parse_ncu_csv(results_dir / "ncu_decode_all.csv")
    flow_data    = parse_ncu_csv(results_dir / "ncu_flow.csv")

    # 타이밍 로드
    # ★ 수정 (260609): timing_results.json 은 두 가지 포맷이 존재
    #   mode_timing 저장 포맷:       {"theoretical":..., "runs":[r1, r2]}
    #   mode_ncu_single_run 포맷:    {"seed":42, "run": r}   ← "runs" 없음!
    #   → runs 없을 때 "run" 키도 시도하도록 수정
    #
    # 또한 ncu 래핑 하에서의 timing (Decode 33분 등 overhead 포함) 은
    # 대역폭 GB/s 계산에 사용하면 안 됨.
    # → ncu 오버헤드 타이밍 검출 시 (decode_total_ms > 100000) 참조값 사용.
    #
    # 참조값 (AppendOnlyCache-C, sdpa, BF16 — 260531 확정):
    TIMING_REF = {
        "VE_ms"              : 728.0,
        "LM_Prefill_ms"      : 1423.0,
        "Decode_total_ms"    : 1503.0,   # 19 steps × 79.1ms (SS)
        "decode_n_steps"     : 19,        # seed=42 실측
        "decode_step_ss_ms"  : 79.1,
        "Flow_ms"            : 870.0,
    }

    timing = {}
    if timing_json.exists():
        with open(timing_json) as f:
            raw = json.load(f)

        # mode_timing 포맷: {"theoretical":..., "runs":[...]}
        runs = raw.get("runs", [])
        if runs:
            timing = runs[-1]
            log.info(f"타이밍 로드 (mode_timing): {timing_json}")
        # mode_ncu_single_run 포맷: {"seed":42, "run":{...}}
        elif "run" in raw:
            timing = raw["run"]
            log.info(f"타이밍 로드 (ncu_single_run overhead): {timing_json}")
            log.warning("  ⚠ ncu 래핑 하 타이밍 — GB/s 계산에 오버헤드 포함됨")
        else:
            log.warning(f"타이밍 파일 포맷 불명: {timing_json}")
    else:
        log.warning(f"타이밍 파일 없음: {timing_json}")

    # ncu 오버헤드 타이밍 검출: Decode > 100초이면 참조값으로 교체
    if timing.get("Decode_total_ms", 0) > 100_000:
        log.warning(
            f"  Decode 타이밍 {timing['Decode_total_ms']/1000:.0f}s → "
            "ncu 오버헤드 값. 참조값(AppendOnlyCache-C 79.1ms/step)으로 대체"
        )
        timing = TIMING_REF
    elif not timing:
        log.warning("  타이밍 없음 → 참조값 사용")
        timing = TIMING_REF

    ve_ms       = timing.get("VE_ms", 0)
    prefill_ms  = timing.get("LM_Prefill_ms", 0)
    decode_ms   = timing.get("Decode_total_ms", 0)
    n_steps     = timing.get("decode_n_steps", 17)
    step_ss_ms  = timing.get("decode_step_ss_ms", 79.1)
    flow_ms     = timing.get("Flow_ms", 0)
    wall_ms     = timing.get("wall_ms", ve_ms + prefill_ms + decode_ms + flow_ms)

    # ── Decode per-step 계산 ─────────────────────────────────────────
    decode_total_gb = decode_data.get("dram_total_GB", 0)
    decode_perstep_gb = (decode_total_gb / n_steps) if n_steps > 0 else 0

    # ── Flow per-step 계산 ───────────────────────────────────────────
    flow_total_gb   = flow_data.get("dram_total_GB", 0)
    flow_perstep_gb = flow_total_gb / FLOW_ODE_STEPS if flow_total_gb > 0 else 0
    flow_step_ms    = flow_ms / FLOW_ODE_STEPS if flow_ms > 0 else 0

    # ── 이론값 (BF16, 전체 가중치 1회 read 기준) ─────────────────────
    # 기존 측정에서 확인된 이론값 (BF16 가중치 크기)
    ve_theory_gb      = 1.879   # VE 가중치 BF16
    prefill_theory_gb = 15.178  # LM 가중치 BF16 (seq>1, compute-bound)
    # Decode 1step: LM가중치 + KV cache
    lm_weights_gb     = 15.178
    seq_tokens        = 3086 + n_steps // 2  # 중간 step 기준
    kv_per_layer_gb   = 2 * seq_tokens * 128 * 2 / 1e9  # K+V, head_dim=128, BF16
    kv_total_gb       = kv_per_layer_gb * 36  # 36 layers
    decode_theory_1step = lm_weights_gb + kv_total_gb
    flow_theory_gb    = 4.561   # Action Expert 가중치 BF16 (1 ODE step)

    # ── 실제 BW 계산 ─────────────────────────────────────────────────
    # ★ BW 비교는 DRAM read 전용으로 계산 (lts__d_sectors_fill_sysmem × 32)
    #   DRAM_PEAK_GBS(231 GB/s)는 단방향 read 피크 기준이므로
    #   (read+write) 합산으로 비교하면 peak% > 100%가 되는 오류 발생
    #   ex) Flow read+write=244.290GB/870ms=280.8GB/s(122%←물리불가)
    #       Flow read only  =176.691GB/870ms=203.1GB/s(88% ←올바름)
    def safe_bw(read_gb, ms_val):
        return f"{bw_gbs(read_gb, ms_val):.1f} GB/s" if read_gb > 0 and ms_val > 0 else "(측정없음)"

    def peak_pct(read_gb, ms_val):
        if read_gb > 0 and ms_val > 0:
            return f"{bw_gbs(read_gb, ms_val)/DRAM_PEAK_GBS*100:.0f}%"
        return "(측정없음)"

    W = 90  # 출력 너비

    print()
    print("═" * W)
    print("  Alpamayo 1.5 on Thor — 4단계 실제 DRAM 대역폭 측정 결과 (260609 수정판)")
    print(f"  SM 11.0 metrics: lts__d_sectors_fill_sysmem × 32 = DRAM read bytes")
    print(f"  DRAM Peak = {DRAM_PEAK_GBS} GB/s (Thor LPDDR5X)")
    print("═" * W)

    # ── VE ────────────────────────────────────────────────────────────
    ve_gb = ve_data.get("dram_total_GB", 0)
    print(f"\n{'─'*W}")
    print(f"  ▌ 1단계: Vision Encoder (VE)")
    print(f"{'─'*W}")
    print(f"  측정 시간  :  {ve_ms:.0f} ms")
    print(f"  이론 DRAM  :  {ve_theory_gb:.3f} GB  (VE 가중치 BF16, 1회 read)")
    ve_read_gb = ve_data.get('dram_read_GB', 0)
    if ve_gb > 0:
        print(f"  실측 DRAM  :  {ve_gb:.3f} GB  "
              f"(read={ve_read_gb:.3f}  write={ve_data.get('dram_write_GB',0):.3f})")
        print(f"  L2 hit rate:  {ve_data.get('l2_hit_pct',0):.1f}%")
        print(f"  read BW    :  {safe_bw(ve_read_gb, ve_ms)}  ({peak_pct(ve_read_gb, ve_ms)} of peak)  [read 기준]")
        print(f"  이론 대비  :  {overhead_ratio(ve_gb, ve_theory_gb)}  "
              f"(eager mode 중간 activation DRAM 접근 포함)")
        print(f"  커널 수    :  {ve_data.get('n_kernels',0):,}")
    else:
        print(f"  실측 DRAM  :  (CSV 없음 또는 파싱 실패)")

    # ── Prefill ───────────────────────────────────────────────────────
    pf_gb = prefill_data.get("dram_total_GB", 0)
    print(f"\n{'─'*W}")
    print(f"  ▌ 2단계: LM Prefill")
    print(f"{'─'*W}")
    print(f"  측정 시간  :  {prefill_ms:.0f} ms")
    print(f"  이론 DRAM  :  {prefill_theory_gb:.3f} GB  (LM 가중치 BF16, seq>>1 → compute-bound)")
    pf_read_gb = prefill_data.get('dram_read_GB', 0)
    if pf_gb > 0:
        print(f"  실측 DRAM  :  {pf_gb:.3f} GB  "
              f"(read={pf_read_gb:.3f}  write={prefill_data.get('dram_write_GB',0):.3f})")
        print(f"  L2 hit rate:  {prefill_data.get('l2_hit_pct',0):.1f}%")
        print(f"  read BW    :  {safe_bw(pf_read_gb, prefill_ms)}  ({peak_pct(pf_read_gb, prefill_ms)} of peak)  [read 기준]")
        print(f"  이론 대비  :  {overhead_ratio(pf_gb, prefill_theory_gb)}  "
              f"(compute-bound: 가중치 재사용으로 DRAM 접근 감소)")
        print(f"  커널 수    :  {prefill_data.get('n_kernels',0):,}")
    else:
        print(f"  실측 DRAM  :  (CSV 없음 또는 파싱 실패)")

    # ── Decode ────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  ▌ 3단계: LM Decode — 전체 EOS까지  ★ 260607 step_010 → 전체 수정")
    print(f"{'─'*W}")
    print(f"  측정 시간  :  {decode_ms:.0f} ms  ({n_steps} steps, SS={step_ss_ms:.1f}ms/step)")
    print(f"  이론 1step :  {decode_theory_1step:.3f} GB  "
              f"(LM={lm_weights_gb:.3f}GB + KV≈{kv_total_gb:.3f}GB, ~mid step)")
    print(f"  이론 전체  :  {decode_theory_1step * n_steps:.3f} GB  ({n_steps} steps)")
    decode_read_gb = decode_data.get('dram_read_GB', 0)
    decode_read_perstep = (decode_read_gb / n_steps) if n_steps > 0 else 0
    if decode_total_gb > 0:
        print(f"  실측 전체  :  {decode_total_gb:.3f} GB  "
              f"(read={decode_read_gb:.3f}  write={decode_data.get('dram_write_GB',0):.3f})")
        print(f"  실측 1step :  {decode_perstep_gb:.3f} GB  ({decode_total_gb:.3f} / {n_steps} steps)")
        print(f"  L2 hit rate:  {decode_data.get('l2_hit_pct',0):.1f}%  (mean, 전체 step)")
        print(f"  read BW    :  {safe_bw(decode_read_gb, decode_ms)}  ({peak_pct(decode_read_gb, decode_ms)} of peak)  [read 기준]")
        print(f"  1step read :  {safe_bw(decode_read_perstep, step_ss_ms)}  ({peak_pct(decode_read_perstep, step_ss_ms)} of peak)")
        print(f"  이론 대비  :  {overhead_ratio(decode_perstep_gb, decode_theory_1step)} per step  "
              f"(eager activation overhead)")
        print(f"  커널 수    :  {decode_data.get('n_kernels',0):,}  (전체 {n_steps} steps)")
    else:
        print(f"  실측 DRAM  :  (CSV 없음 또는 파싱 실패)")
        print(f"  ⚠ 260607 step_010 데이터: 16.980 GB (단 1 step, 전체 아님)")
        print(f"  ⚠ 이번 측정으로 전체 {n_steps} steps 확인 필요")

    # ── Flow ──────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  ▌ 4단계: Flow ODE (Action Expert, {FLOW_ODE_STEPS} steps)")
    print(f"{'─'*W}")
    print(f"  측정 시간  :  {flow_ms:.0f} ms  ({FLOW_ODE_STEPS} steps × {flow_step_ms:.0f}ms/step)")
    print(f"  이론 1step :  {flow_theory_gb:.3f} GB  (Action Expert 가중치 BF16)")
    print(f"  이론 전체  :  {flow_theory_gb * FLOW_ODE_STEPS:.3f} GB  ({FLOW_ODE_STEPS} steps)")
    flow_read_gb = flow_data.get('dram_read_GB', 0)
    flow_read_perstep = flow_read_gb / FLOW_ODE_STEPS if flow_read_gb > 0 else 0
    if flow_total_gb > 0:
        print(f"  실측 전체  :  {flow_total_gb:.3f} GB  "
              f"(read={flow_read_gb:.3f}  write={flow_data.get('dram_write_GB',0):.3f})")
        print(f"  실측 1step :  {flow_perstep_gb:.3f} GB  ({flow_total_gb:.3f} / {FLOW_ODE_STEPS} steps)")
        print(f"  L2 hit rate:  {flow_data.get('l2_hit_pct',0):.1f}%  (mean, 전체 {FLOW_ODE_STEPS} steps)")
        print(f"  read BW    :  {safe_bw(flow_read_gb, flow_ms)}  ({peak_pct(flow_read_gb, flow_ms)} of peak)  [read 기준]")
        print(f"  1step read :  {safe_bw(flow_read_perstep, flow_step_ms)}  ({peak_pct(flow_read_perstep, flow_step_ms)} of peak)")
        print(f"  write rate :  {flow_data.get('dram_write_GB',0):.3f} GB total  "
              f"({flow_data.get('dram_write_GB',0)/flow_total_gb*100:.0f}% of total traffic)  "
              f"← eager mode L2 eviction+activation write-back")
        print(f"  이론 대비  :  {overhead_ratio(flow_perstep_gb, flow_theory_gb)} per step  "
              f"(eager activation overhead, write-back 포함)")
        print(f"  커널 수    :  {flow_data.get('n_kernels',0):,}  (전체 {FLOW_ODE_STEPS} steps)")
    else:
        print(f"  실측 DRAM  :  (CSV 없음 또는 파싱 실패)")
        # 260607 실측값 참고
        flow_ref = 122.114
        print(f"  참고 260607: {flow_ref:.3f} GB 전체, {flow_ref/FLOW_ODE_STEPS:.3f} GB/step (22.0% L2 hit)")

    # ── 전체 요약 ─────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  ▌ 전체 요약")
    print(f"{'─'*W}")

    total_actual = sum(filter(None, [
        ve_gb or 0,
        pf_gb or 0,
        decode_total_gb or 0,
        flow_total_gb or 0,
    ]))

    print(f"  {'단계':<16} {'시간':>8}   {'총DRAM(R+W)':>13}   {'read BW':>12}   {'Peak%':>7}   {'L2 hit':>7}")
    print(f"  {'─'*16} {'─'*8}   {'─'*13}   {'─'*12}   {'─'*7}   {'─'*7}")

    def row(name, ms_val, total_gb, read_gb, l2):
        tot_s = f"{total_gb:.3f} GB" if total_gb > 0 else "—"
        bw_s  = f"{bw_gbs(read_gb, ms_val):.1f} GB/s"  if read_gb > 0 and ms_val > 0 else "—"
        pk_s  = peak_pct(read_gb, ms_val) if read_gb > 0 else "—"
        l2_s  = f"{l2:.1f}%" if l2 > 0 else "—"
        return f"  {name:<16} {ms_val:>7.0f}ms   {tot_s:>13}   {bw_s:>12}   {pk_s:>7}   {l2_s:>7}"

    print(row("VE",          ve_ms,      ve_gb or 0,          ve_read_gb or 0,      ve_data.get("l2_hit_pct",0)))
    print(row("LM Prefill",  prefill_ms, pf_gb or 0,          pf_read_gb or 0,      prefill_data.get("l2_hit_pct",0)))
    print(row("Decode(all)", decode_ms,  decode_total_gb or 0, decode_read_gb or 0, decode_data.get("l2_hit_pct",0)))
    print(row(f"Flow(×{FLOW_ODE_STEPS}ODE)", flow_ms, flow_total_gb or 0, flow_read_gb or 0, flow_data.get("l2_hit_pct",0)))
    print(f"  {'─'*16} {'─'*8}   {'─'*12}   {'─'*12}   {'─'*7}   {'─'*7}")
    print(f"  {'TOTAL':<16} {wall_ms:>7.0f}ms   {total_actual:>8.3f} GB")

    print()

    # ── Decode per-step 상세 ──────────────────────────────────────────
    if decode_total_gb > 0 and n_steps > 0:
        print(f"  ▌ Decode 1 step 상세")
        print(f"  실측 1step DRAM : {decode_perstep_gb:.3f} GB")
        print(f"   ├─ 이론 LM 가중치: {lm_weights_gb:.3f} GB  (BF16, 1회 read)")
        print(f"   ├─ 이론 KV cache : {kv_total_gb:.3f} GB  (~mid step)")
        print(f"   ├─ 이론 합계     : {decode_theory_1step:.3f} GB")
        print(f"   └─ 실측/이론     : {overhead_ratio(decode_perstep_gb, decode_theory_1step)}  "
              f"(eager activation 포함)")
        print()

    # ── JSON 저장 ─────────────────────────────────────────────────────
    out = {
        "meta": {
            "script": "260609_analyze_ncu_full.py",
            "dram_peak_gbs": DRAM_PEAK_GBS,
            "sector_bytes": SECTOR_BYTES,
            "flow_ode_steps": FLOW_ODE_STEPS,
        },
        "timing": timing,
        "VE":          {**ve_data,      "bw_gbs": round(bw_gbs(ve_gb or 0, ve_ms), 1)},
        "LM_Prefill":  {**prefill_data, "bw_gbs": round(bw_gbs(pf_gb or 0, prefill_ms), 1)},
        "Decode_all":  {
            **decode_data,
            "n_steps":           n_steps,
            "dram_per_step_GB":  round(decode_perstep_gb, 3),
            "bw_gbs_total":      round(bw_gbs(decode_total_gb or 0, decode_ms), 1),
            "bw_gbs_per_step":   round(bw_gbs(decode_perstep_gb, step_ss_ms), 1),
            "theory_per_step_GB": round(decode_theory_1step, 3),
        },
        "Flow": {
            **flow_data,
            "ode_steps":          FLOW_ODE_STEPS,
            "dram_per_step_GB":   round(flow_perstep_gb, 3),
            "bw_gbs_total":       round(bw_gbs(flow_total_gb or 0, flow_ms), 1),
            "bw_gbs_per_step":    round(bw_gbs(flow_perstep_gb, flow_step_ms), 1),
            "theory_per_step_GB": round(flow_theory_gb, 3),
        },
    }

    out_path = results_dir / "bandwidth_analysis_full.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info(f"분석 결과 저장: {out_path}")

    print("═" * W)
    print(f"  JSON 저장: {out_path}")
    print("═" * W)
    print()


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Alpamayo 4단계 DRAM 분석 (260609 수정판)")
    p.add_argument("--results-dir",
                   default="/home/ice401/alpamayo1.5/profiling_results/260609_ncu_full")
    p.add_argument("--timing-json", default="")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    timing_json = Path(args.timing_json) if args.timing_json \
                  else results_dir / "timing_results.json"

    analyze(results_dir, timing_json)


if __name__ == "__main__":
    main()
