"""Module-level fusion injection — zero model modification.

Instead of rewriting checkpoints or model source, UMIC swaps the *forward*
of matching submodules at load time. The match is structural (duck-typed),
not class-based, so it survives model version changes: any module exposing
`gate_proj` / `up_proj` / `down_proj` Linears with a SiLU-family activation
is a P5 candidate — Qwen2, Qwen3, Llama, and whatever Alpamayo 2.0 ships,
as long as the motif is present.

Weights are shared (no copy, no quantization, no value change); only the
execution schedule of the same math changes.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from umic.kernels.fused_ffn import gate_silu_mul

logger = logging.getLogger(__name__)

_SILU_NAMES = ("silu", "swish")

# Below this row count the motif runs as GEMV (decode, seq=1) where cuBLAS
# is optimal and the eager intermediates are KB-scale — fusion would only
# hurt. Regime-aware dispatch, design doc 원칙 2.
FUSE_MIN_ROWS = 64


def _is_p5_mlp(module: nn.Module) -> bool:
    """Structural match for the gate/up/down SiLU MLP motif (pattern P5)."""
    for attr in ("gate_proj", "up_proj", "down_proj"):
        sub = getattr(module, attr, None)
        if not isinstance(sub, nn.Linear) or sub.bias is not None:
            return False
    act = getattr(module, "act_fn", None) or getattr(module, "act", None)
    act_name = type(act).__name__.lower() if act is not None else ""
    return any(s in act_name for s in _SILU_NAMES)


def _fused_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """P5-fused replacement: silu(x@Wg)*(x@Wu) in one kernel, then down."""
    shape = x.shape
    x2d = x.reshape(-1, shape[-1])
    if x2d.shape[0] < FUSE_MIN_ROWS:
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(h)
    # nn.Linear stores weight as [out, in]; the kernel takes [in, out]
    # strides, so .t() is a free view — no copy, weights untouched.
    h = gate_silu_mul(x2d, self.gate_proj.weight.t(), self.up_proj.weight.t())
    out = self.down_proj(h)
    return out.reshape(*shape[:-1], out.shape[-1])


def fuse_mlps(model: nn.Module, dry_run: bool = False) -> int:
    """Swap the forward of every P5-matching MLP in `model`.

    Args:
        model: Any nn.Module tree (unmodified checkpoint).
        dry_run: If True, only count matches without patching.

    Returns:
        Number of modules matched (and patched unless dry_run).
    """
    count = 0
    for name, module in model.named_modules():
        if _is_p5_mlp(module):
            count += 1
            if not dry_run:
                module.forward = _fused_mlp_forward.__get__(module)
                logger.info("P5 fused: %s", name)
    logger.info("fuse_mlps: %d module(s) %s", count,
                "matched (dry run)" if dry_run else "patched")
    return count


def unfuse_mlps(model: nn.Module) -> int:
    """Restore original forwards (delete instance overrides)."""
    count = 0
    for _, module in model.named_modules():
        if "forward" in module.__dict__:
            del module.__dict__["forward"]
            count += 1
    return count
