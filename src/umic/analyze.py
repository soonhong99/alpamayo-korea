"""Bytes-moved static analyzer — the cost model that drives every decision.

For each node in an fx graph, estimate DRAM read/write bytes at the kernel
boundary (eager semantics: every node writes its output to DRAM and the
consumer reads it back). The per-graph total is validated against ncu
hardware counters (lts__d_sectors_fill_sysmem); fusion candidates are then
ranked by predicted byte savings.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Iterable

import torch
import torch.fx

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class NodeTraffic:
    """Estimated DRAM traffic for one kernel boundary."""

    node: str
    op: str
    read_bytes: int
    write_bytes: int

    @property
    def total(self) -> int:
        return self.read_bytes + self.write_bytes


def _tensor_bytes(meta_val) -> int:
    """Bytes of a tensor (or pytree of tensors) from fx meta['val']."""
    if isinstance(meta_val, torch.Tensor):
        return meta_val.numel() * meta_val.element_size()
    if isinstance(meta_val, (list, tuple)):
        return sum(_tensor_bytes(v) for v in meta_val)
    return 0


def analyze_graph(gm: torch.fx.GraphModule) -> list[NodeTraffic]:
    """Estimate per-node DRAM traffic under eager (non-fused) execution.

    Requires shape propagation: run
    `torch.fx.passes.shape_prop.ShapeProp(gm).propagate(*example_inputs)`
    first so every node carries meta['val'] / meta['tensor_meta'].

    Returns:
        One NodeTraffic per compute node, in graph order.
    """
    results: list[NodeTraffic] = []
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output", "get_attr"):
            continue
        out_bytes = _tensor_bytes(node.meta.get("val"))
        in_bytes = sum(
            _tensor_bytes(arg.meta.get("val"))
            for arg in node.all_input_nodes
        )
        results.append(
            NodeTraffic(node=node.name, op=str(node.target),
                        read_bytes=in_bytes, write_bytes=out_bytes)
        )
    return results


def total_gb(traffic: Iterable[NodeTraffic]) -> float:
    """Sum traffic in GB (decimal, matching ncu reporting)."""
    return sum(t.total for t in traffic) / 1e9


def compare_with_ncu(predicted_gb: float, measured_gb: float) -> float:
    """Return prediction error ratio; log a calibration verdict.

    The M0 gate is |error| <= 0.30 on LM Prefill (measured 231.97 GB).
    """
    err = (predicted_gb - measured_gb) / measured_gb
    logger.info(
        "bytes-moved model: predicted %.1f GB, ncu measured %.1f GB, error %+.1f%%",
        predicted_gb, measured_gb, 100 * err,
    )
    return err
