"""Declarative fusion pattern registry (the closed set, ~10 motifs).

A pattern is a named sequence of ATen-op predicates plus an estimated
byte-savings function. Adding support for a new model family means adding
patterns here — never touching the engine core. Unmatched subgraphs run
eager; coverage never blocks correctness.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Sequence

import torch.fx

logger = logging.getLogger(__name__)

# A predicate decides whether an fx node is the next op of the motif.
NodePred = Callable[[torch.fx.Node], bool]


def _is(*target_names: str) -> NodePred:
    """Predicate: node.target string contains any of the given names."""
    def pred(node: torch.fx.Node) -> bool:
        t = str(node.target)
        return any(name in t for name in target_names)
    return pred


@dataclasses.dataclass(frozen=True)
class FusionPattern:
    """One fusable motif.

    Attributes:
        name: Registry key, also the fused-kernel registry key (L0).
        ops: Ordered op predicates forming a linear chain.
        regime: "A" (compute-limited stages) or "B" (memory-saturated).
        note: Where the savings come from.
    """

    name: str
    ops: Sequence[NodePred]
    regime: str
    note: str


REGISTRY: dict[str, FusionPattern] = {}


def register(p: FusionPattern) -> None:
    REGISTRY[p.name] = p


# --- The transformer motif set (design doc §1, 원칙 3) -------------------

register(FusionPattern(
    "gate_silu_mul",
    ops=(_is("linear", "mm"), _is("silu"), _is("mul")),
    regime="A",
    note="FFN gate/up + SiLU + elementwise mul in one kernel; biggest "
         "prefill saving (~30 GB): the [seq, 11008] intermediates never "
         "touch DRAM.",
))
register(FusionPattern(
    "norm_proj",
    ops=(_is("rms_norm", "layer_norm", "native_layer_norm"), _is("linear", "mm")),
    regime="A",
    note="Norm output consumed in-register by the projection (~15 GB).",
))
register(FusionPattern(
    "qkv_rope",
    ops=(_is("linear", "mm"), _is("rope", "rotary", "mul")),
    regime="A",
    note="QKV projection + rotary embedding without Q/K round-trips.",
))
register(FusionPattern(
    "o_proj_residual",
    ops=(_is("linear", "mm"), _is("add")),
    regime="A",
    note="Output projection epilogue adds the residual in-kernel.",
))
register(FusionPattern(
    "ode_step",
    ops=(_is("mul"), _is("add")),
    regime="B",
    note="x <- x + dt*v inside the flow persistent kernel (M2).",
))
register(FusionPattern(
    "adaln",
    ops=(_is("native_layer_norm", "layer_norm"), _is("mul"), _is("add")),
    regime="B",
    note="DiT AdaLN scale/shift fused for the Action Expert.",
))


def match(gm: torch.fx.GraphModule) -> list[tuple[FusionPattern, list[torch.fx.Node]]]:
    """Greedy linear-chain matcher over the graph in topological order.

    Deliberately simple (M1): matches single-consumer chains only, which
    covers the transformer motifs above. DAG-shaped patterns (qkv 3-way
    split) get a dedicated matcher later if profiling justifies it.

    Returns:
        (pattern, nodes) for every non-overlapping match.
    """
    matches: list[tuple[FusionPattern, list[torch.fx.Node]]] = []
    claimed: set[torch.fx.Node] = set()

    nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    for i, head in enumerate(nodes):
        if head in claimed:
            continue
        for pattern in REGISTRY.values():
            chain = _try_chain(head, pattern.ops, claimed)
            if chain:
                matches.append((pattern, chain))
                claimed.update(chain)
                break
    logger.info("pattern matches: %s",
                {p.name: sum(1 for q, _ in matches if q is p) for p in REGISTRY.values()})
    return matches


def _try_chain(head: torch.fx.Node, preds: Sequence[NodePred],
               claimed: set) -> list[torch.fx.Node] | None:
    """Follow single-consumer edges matching each predicate in order."""
    if not preds[0](head):
        return None
    chain = [head]
    cur = head
    for pred in preds[1:]:
        users = [u for u in cur.users if u.op == "call_function"]
        if len(users) != 1 or users[0] in claimed or not pred(users[0]):
            return None
        cur = users[0]
        chain.append(cur)
    return chain
