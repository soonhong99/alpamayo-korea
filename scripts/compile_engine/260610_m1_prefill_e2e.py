"""M1 step 2: P5 fusion injected into the REAL Alpamayo LM, end-to-end.

Thin wrapper around scripts/profiling/260609_ncu_full_bandwidth.py (imported
unmodified) that adds one thing: `--fuse 1` applies umic.integrate.fuse_mlps
to the LM module after loading. Everything else — model loading, NVTX phase
separation, timing/ncu modes — is reused as-is.

Eager baseline is the confirmed 260609 measurement (Prefill 231.97 GB),
so only the fused run needs ncu.

Usage on Thor:
  PYTHONPATH=~/alpamayo1.5/src python3 260610_m1_prefill_e2e.py --fuse 1 --mode timing
  (ncu): see 260610_run_ncu_m1_prefill.sh
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

log = logging.getLogger("umic.m1e2e")

BASE_PATH = Path.home() / "alpamayo1.5/scripts/profiling/260609_ncu_full_bandwidth.py"
RESULTS_DIR = Path.home() / "alpamayo1.5/profiling_results/260610_m1_prefill_e2e"


def load_base_module():
    """Import 260609_ncu_full_bandwidth.py (digit-leading name needs spec)."""
    spec = importlib.util.spec_from_file_location("ncu_full_bandwidth", BASE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ncu_full_bandwidth"] = mod
    spec.loader.exec_module(mod)
    return mod


def find_lm_module(model):
    """Locate the LM trunk the same way the base script does."""
    for attr in ("language_model", "model"):
        cand = getattr(model.vlm, attr, None)
        if cand is None:
            continue
        if hasattr(cand, "layers"):
            return cand
        sub = getattr(cand, "model", None)
        if sub is not None and hasattr(sub, "layers"):
            return cand
    raise RuntimeError("LM module not found")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["timing", "ncu_single_run"], default="timing")
    p.add_argument("--fuse", type=int, default=1)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = p.parse_args()

    base = load_base_module()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = base.load_model()

    if args.fuse:
        from umic.integrate import fuse_mlps
        lm = find_lm_module(model)
        n = fuse_mlps(lm)
        log.info("UMIC P5: %d LM MLP(s) patched (seq<64 dispatches to eager)", n)
        assert n > 0, "no P5 match in LM — structural motif changed?"

    data, helper = base.load_raw_data()
    model_inputs = base.prepare_model_inputs(model, data, helper)

    if args.mode == "timing":
        base.mode_timing(model, model_inputs, results_dir)
    else:
        base.mode_ncu_single_run(model, model_inputs, results_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    main()
