"""
260522_model_cache_fit.py — Alpamayo 1.5 Weight Footprint vs GPU Cache
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Loads the full Alpamayo 1.5 model and reports per-layer weight sizes,
then compares them against the measured GPU cache sizes:
  - GPU L2    : 32 MB
  - SMEM/SM   : 228 KB  (20 SMs total)

Also calculates KV-cache footprint for the decode phase and checks
whether it could fit in L2 — this is the closest thing to
"fitting model state in cache" that's physically achievable.

Key findings expected:
  - No single transformer layer fits in L2 (individual Q/K/V projections ≥ 33 MB)
  - KV cache for 65 decode steps is ~same magnitude as L2 → interesting edge case
  - Smaller Action Expert layers might partially fit

Run on Thor (requires a1_5_venv, model loading ~3-4 min):
    source ~/alpamayo1.5/a1_5_venv/bin/activate
    python3 ~/alpamayo1.5/scripts/cache_experiments/260522_model_cache_fit.py

Output: ~/alpamayo1.5/profiling_results/260522_gpu_cache/model_cache_fit.json
"""

import json
import sys
from pathlib import Path

import torch

# From hardware_spec.json (previously measured)
GPU_L2_BYTES = 33_554_432        # 32 MB exactly
GPU_SMEM_PER_SM_KB = 228.0
GPU_N_SM = 20
DECODE_STEPS = 65               # from profiling_results: Alpamayo uses 65 decode tokens


def fmt(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024**2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def module_own_param_bytes(module: torch.nn.Module) -> int:
    return (
        sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
        + sum(b.numel() * b.element_size() for b in module.buffers(recurse=False))
    )


def collect_linear_layers(model: torch.nn.Module) -> list[dict]:
    """
    Collects all nn.Linear (and similar) layers with their weight shapes.
    These are the weight matrices accessed during inference.
    """
    rows = []
    for name, module in model.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        w = module.weight
        size_bytes = w.numel() * w.element_size()
        rows.append({
            "name": name,
            "class": type(module).__name__,
            "weight_shape": list(w.shape),
            "size_bytes": size_bytes,
            "size_human": fmt(size_bytes),
            "fits_l2": size_bytes <= GPU_L2_BYTES,
            "fits_smem_per_sm": size_bytes <= int(GPU_SMEM_PER_SM_KB * 1024),
        })
    return sorted(rows, key=lambda r: r["size_bytes"], reverse=True)


def estimate_kv_cache_bytes(n_layers: int, n_heads: int, d_head: int, seq_len: int, dtype_bytes: int = 2) -> int:
    # K cache: [n_layers, seq_len, n_heads, d_head]
    # V cache: same
    return 2 * n_layers * seq_len * n_heads * d_head * dtype_bytes


def detect_architecture(rows: list[dict]) -> dict:
    """
    Infers likely n_layers, d_model, n_heads from the collected linear layers.
    Looks for q_proj or similar weight shapes.
    """
    info: dict = {}
    for r in rows:
        name = r["name"].lower()
        shape = r["weight_shape"]
        if len(shape) == 2 and ("q_proj" in name or "query" in name):
            out_dim, in_dim = shape
            info.setdefault("d_model_candidates", set()).add(max(out_dim, in_dim))
    return info


def main() -> None:
    # Package lives at ~/alpamayo1.5/src/alpamayo1_5/
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    print("Loading Alpamayo 1.5 (BF16, CPU only for counting) ...")
    # Load to CPU first — faster for weight counting; no GPU VRAM needed
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    print(f"\nModel       : Alpamayo 1.5")
    print(f"Total params: {total_params / 1e9:.2f}B")
    print(f"Total size  : {fmt(total_bytes)}  (BF16)")
    print(f"GPU L2      : {fmt(GPU_L2_BYTES)}")
    print(f"SMEM/SM     : {GPU_SMEM_PER_SM_KB:.0f} KB")
    print(f"Model/L2    : {total_bytes / GPU_L2_BYTES:.0f}× larger than L2")

    rows = collect_linear_layers(model)
    n_fit_l2 = sum(1 for r in rows if r["fits_l2"])
    n_fit_smem = sum(1 for r in rows if r["fits_smem_per_sm"])

    print(f"\nLinear layers total         : {len(rows)}")
    print(f"  fitting in L2 (32 MB)     : {n_fit_l2}")
    print(f"  fitting in SMEM/SM (228KB): {n_fit_smem}")

    # Print top-30 largest
    print(f"\n{'Rank':>4}  {'Layer name':<55}  {'Shape':<20}  {'Size':>10}  {'L2?':>5}")
    print("─" * 105)
    for i, r in enumerate(rows[:30], 1):
        shape_str = "×".join(str(x) for x in r["weight_shape"])
        flag = "YES" if r["fits_l2"] else "  -"
        print(f"{i:>4}  {r['name']:<55}  {shape_str:<20}  {r['size_human']:>10}  {flag:>5}")

    # Print smallest layers that do fit
    fitting = [r for r in rows if r["fits_l2"]]
    if fitting:
        print(f"\n── Layers that fit in L2 ({len(fitting)} total, largest 10 shown) ──")
        for r in fitting[:10]:
            shape_str = "×".join(str(x) for x in r["weight_shape"])
            print(f"  {r['name']:<55}  {shape_str:<20}  {r['size_human']:>10}")

    # KV cache analysis
    print(f"\n── KV Cache Footprint for {DECODE_STEPS} Decode Steps ──")
    print(f"{'Scenario':<35}  {'KV size':>10}  {'vs L2':>12}  {'Fits?':>6}")
    print("─" * 70)

    # Try to estimate from actual weight shapes
    # Look for patterns in rows to infer n_layers, n_heads, d_head
    kv_scenarios = [
        ("Cosmos Reason2 LM (estimate)",     28, 32, 128, DECODE_STEPS),
        ("Cosmos Reason2 LM GQA-4 (estim.)", 28, 8,  128, DECODE_STEPS),
        ("Action Expert (estimate)",         16, 16, 128, DECODE_STEPS),
        ("Action Expert GQA-2 (estimate)",   16, 8,  128, DECODE_STEPS),
    ]
    kv_results = []
    for label, n_layers, n_heads, d_head, steps in kv_scenarios:
        kv_bytes = estimate_kv_cache_bytes(n_layers, n_heads, d_head, steps)
        ratio = kv_bytes / GPU_L2_BYTES
        fits = "YES" if ratio <= 1.0 else "NO"
        print(f"  {label:<33}  {fmt(kv_bytes):>10}  {ratio:>11.2f}×  {fits:>6}")
        kv_results.append({
            "label": label,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_head": d_head,
            "decode_steps": steps,
            "kv_bytes": kv_bytes,
            "kv_mb": kv_bytes / 1024 ** 2,
            "ratio_vs_l2": ratio,
        })

    # Save
    out_dir = Path("profiling_results/260522_gpu_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_cache_fit.json"
    out_path.write_text(json.dumps({
        "hardware": {
            "gpu_l2_bytes": GPU_L2_BYTES,
            "gpu_l2_mb": GPU_L2_BYTES / 1024 ** 2,
            "smem_per_sm_kb": GPU_SMEM_PER_SM_KB,
            "n_sm": GPU_N_SM,
        },
        "model": {
            "total_params_B": total_params / 1e9,
            "total_bytes": total_bytes,
            "total_gb": total_bytes / 1024 ** 3,
            "model_to_l2_ratio": total_bytes / GPU_L2_BYTES,
        },
        "linear_layers": rows,
        "kv_cache_analysis": kv_results,
    }, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
