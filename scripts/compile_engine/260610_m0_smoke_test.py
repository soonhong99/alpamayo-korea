"""UMIC M0 smoke test — run ON THOR inside the a1_5 venv.

Gates (design doc §5, M0):
  1. Triton 3.7.0 runs standalone @triton.jit kernels on SM 11.0
     (torch.compile/Inductor is known-broken; this tests the runtime alone).
  2. Fused gate_silu_mul matches eager within bf16 tolerance.
  3. Timing comparison at prefill shape (M=3086, K=4096, N=11008).

Usage (on Thor):
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  cd ~/alpamayo1.5 && PYTHONPATH=src python3 scripts/compile_engine/260610_m0_smoke_test.py
"""

from __future__ import annotations

import logging
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("umic.m0")


def gate1_triton_runtime() -> bool:
    """Standalone Triton vector-add on SM 11.0."""
    try:
        import triton
        import triton.language as tl
    except Exception as exc:  # noqa: BLE001
        log.error("GATE1 FAIL: triton import: %s", exc)
        return False

    @triton.jit
    def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(o_ptr + offs,
                 tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
                 mask=mask)

    try:
        n = 1 << 20
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        o = torch.empty_like(x)
        _add[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
        torch.cuda.synchronize()
        ok = torch.allclose(o, x + y)
        log.info("GATE1 %s: triton %s vector-add on %s",
                 "PASS" if ok else "FAIL", triton.__version__,
                 torch.cuda.get_device_name())
        return ok
    except Exception as exc:  # noqa: BLE001
        log.error("GATE1 FAIL: kernel launch/compile: %s", exc)
        return False


def gate2_fused_ffn_correctness() -> bool:
    """Fused gate_silu_mul vs eager, bf16."""
    from umic.kernels.fused_ffn import HAS_TRITON, gate_silu_mul_eager
    if not HAS_TRITON:
        log.error("GATE2 SKIP: no triton")
        return False
    from umic.kernels.fused_ffn import gate_silu_mul_triton

    torch.manual_seed(0)
    M, K, N = 128, 256, 512
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    wg = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.05
    wu = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.05

    ref = gate_silu_mul_eager(x.float(), wg.float(), wu.float())
    got = gate_silu_mul_triton(x, wg, wu).float()
    err = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    ok = err < 1e-2
    log.info("GATE2 %s: relative max error %.2e (gate < 1e-2)",
             "PASS" if ok else "FAIL", err)
    return ok


def gate3_prefill_shape_timing() -> None:
    """Eager (4 kernels) vs fused, at the real prefill FFN shape."""
    from umic.kernels.fused_ffn import HAS_TRITON, gate_silu_mul_eager
    if not HAS_TRITON:
        return
    from umic.kernels.fused_ffn import gate_silu_mul_triton

    M, K, N = 3086, 4096, 11008
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    wg = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    wu = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    def bench(fn, iters: int = 10) -> float:
        for _ in range(3):
            fn(x, wg, wu)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(x, wg, wu)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    log.info("GATE3 timing @ [%d,%d]x[%d,%d]: eager %.2f ms | fused %.2f ms",
             M, K, K, N, bench(gate_silu_mul_eager), bench(gate_silu_mul_triton))
    log.info("GATE3 note: per-call DRAM saving expected = 2 intermediates "
             "round-trip = %.0f MB", 2 * 2 * M * N * 2 / 1e6)


def gate4_pipeline_ir() -> None:
    """Pipeline IR + prefetch scheduler dry run (no model needed)."""
    from umic import Pipeline, Repeat, Stage

    ident = torch.nn.Identity()
    pipe = Pipeline("alpamayo15_dry", stages=[
        Stage("ve", ident, weights_gb=1.153),
        Stage("prefill", ident, weights_gb=15.168, mode="prefill"),
        Repeat(Stage("decode", ident, weights_gb=15.168, mode="decode"),
               until="eos", max_iter=64),
        Repeat(Stage("flow", ident, weights_gb=4.561), times=65),
    ])
    engine = pipe.compile()
    for a in engine.schedule:  # type: ignore[attr-defined]
        log.info("GATE4 schedule: prefetch %s during %s (%.1f ms)",
                 a.weights_of, a.during, a.transfer_ms)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "run this on Thor"
    g1 = gate1_triton_runtime()
    if g1:
        gate2_fused_ffn_correctness()
        gate3_prefill_shape_timing()
    gate4_pipeline_ir()
    log.info("M0 smoke test done. If GATE1 failed -> plan B: CUDA C++ "
             "kernels via torch.utils.cpp_extension (nvcc 13.0).")
