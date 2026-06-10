"""M1 step 1: P5 fusion on a standalone Qwen-style MLP, ncu byte validation.

No model loading needed: the byte traffic of the FFN motif depends only on
shapes ([3086, 4096] x inter 11008, bf16 — the real Alpamayo prefill shape),
so a randomly initialised module measures the same DRAM behaviour in
seconds instead of minutes.

Modes:
  bench               correctness + wall-clock, eager vs fused (default)
  ncu_eager           one NVTX-wrapped eager forward (for ncu)
  ncu_fused           one NVTX-wrapped fused forward (for ncu)
  summarize <csv...>  sum DRAM read/write bytes from ncu CSV output

Usage on Thor: see scripts/compile_engine/260610_run_ncu_m1.sh
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("umic.m1")

# Real Alpamayo LM prefill FFN shape (docs/2606_1주차/260608_02 §1.4).
SEQ, HIDDEN, INTER = 3086, 4096, 11008


def build_mlp():
    """Qwen2/Llama-style gate/up/down SiLU MLP, bf16, random weights."""
    import torch
    from torch import nn

    class Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(HIDDEN, INTER, bias=False)
            self.up_proj = nn.Linear(HIDDEN, INTER, bias=False)
            self.down_proj = nn.Linear(INTER, HIDDEN, bias=False)
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    torch.manual_seed(0)
    return Mlp().to("cuda", dtype=torch.bfloat16).eval()


def mode_bench() -> None:
    import torch
    from umic.integrate import fuse_mlps

    mlp = build_mlp()
    x = torch.randn(SEQ, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1

    with torch.no_grad():
        ref = mlp(x).float()
        n = fuse_mlps(mlp)
        assert n == 1, f"expected 1 P5 match, got {n}"
        got = mlp(x).float()
    err = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    log.info("correctness: relative max error %.2e (gate < 1e-2): %s",
             err, "PASS" if err < 1e-2 else "FAIL")

    def bench(fn, iters: int = 10) -> float:
        with torch.no_grad():
            for _ in range(3):
                fn(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(x)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    fused_ms = bench(mlp)
    from umic.integrate import unfuse_mlps
    unfuse_mlps(mlp)
    eager_ms = bench(mlp)
    log.info("full MLP forward: eager %.2f ms | fused(P5)+down %.2f ms",
             eager_ms, fused_ms)


def mode_ncu(fused: bool) -> None:
    import torch
    from umic.integrate import fuse_mlps

    mlp = build_mlp()
    x = torch.randn(SEQ, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
    if fused:
        fuse_mlps(mlp)
    with torch.no_grad():
        mlp(x)  # warmup: JIT/cuBLAS heuristics outside the NVTX range
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("Phase")
        torch.cuda.nvtx.range_push("MLP")
        mlp(x)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    log.info("ncu run done (%s)", "fused" if fused else "eager")


def mode_summarize(paths: list[str]) -> None:
    """Sum DRAM read/write from ncu --csv per-kernel output (sectors x 32B)."""
    for path in paths:
        read_b = write_b = 0
        kernels = set()
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            # ncu prepends log lines; skip until the CSV header row.
            rows = [ln for ln in f if ln.startswith('"')]
        for rec in csv.DictReader(rows):
            metric = rec.get("Metric Name", "")
            try:
                val = float(rec.get("Metric Value", "0").replace(",", ""))
            except ValueError:
                continue
            kernels.add((rec.get("ID"), rec.get("Kernel Name")))
            if metric == "lts__d_sectors_fill_sysmem.sum":
                read_b += val * 32
            elif metric == "lts__t_sectors_aperture_sysmem_op_write.sum":
                write_b += val * 32
        print(f"{path}: kernels={len(kernels)} "
              f"read={read_b / 1e6:.1f} MB write={write_b / 1e6:.1f} MB "
              f"total={(read_b + write_b) / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="bench",
                    choices=["bench", "ncu_eager", "ncu_fused", "summarize"])
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()

    if args.mode == "summarize":
        mode_summarize(args.files)
        sys.exit(0)
    import torch
    assert torch.cuda.is_available(), "run on Thor"
    if args.mode == "bench":
        mode_bench()
    else:
        mode_ncu(fused=args.mode == "ncu_fused")
