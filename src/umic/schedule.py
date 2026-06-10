"""Pipeline-IR scheduler: cross-stage prefetch under a shared-bus BW budget.

The iGPU constraint that makes this interesting: DMA prefetch and SM compute
share one LPDDR5X bus. A prefetch is only scheduled where measured headroom
(peak BW minus the stage's own demand) can complete the transfer before the
weights are needed. The decode-stage numbers below come straight from ncu
measurement (docs/2606_1주차/260608_01): decode reads at 89% of peak, so
per-layer prefetch is impossible — only stage-granularity prefetch fits.
"""

from __future__ import annotations

import dataclasses
import logging

from umic.hw import HardwareProfile

logger = logging.getLogger(__name__)

# Measured read-BW utilisation per stage regime (260609 ncu, fraction of peak).
# Used as default demand when no profile_dir is given; a profile overrides.
DEFAULT_BW_UTIL = {"ve": 0.35, "prefill": 0.55, "decode": 0.89, "flow": 0.88}


@dataclasses.dataclass
class PrefetchAction:
    """One scheduled DMA prefetch.

    Attributes:
        weights_of: Stage whose weights are being prefetched.
        during: Stage during which the DMA runs.
        start_frac: Fraction of `during`'s runtime at which DMA starts.
        transfer_ms: Expected transfer time at the available headroom BW.
    """

    weights_of: str
    during: str
    start_frac: float
    transfer_ms: float


def plan_cross_stage_prefetch(pipeline, hw: HardwareProfile) -> list[PrefetchAction]:
    """Greedily schedule each stage's weights into the previous stage's headroom.

    For consecutive stages (A -> B): headroom = peak * (1 - util_A).
    If B's weights transfer within A's remaining runtime, schedule the DMA
    as late as possible (LIFO with the LRU working set: late prefetch keeps
    L2/DRAM-controller pressure away from A's critical early phase).

    Args:
        pipeline: The Pipeline (uses flat_stages and stage durations if known).
        hw: Hardware profile providing the shared-bus peak BW.

    Returns:
        List of PrefetchAction; empty entries mean "load on demand".
    """
    actions: list[PrefetchAction] = []
    stages = pipeline.flat_stages()
    for prev, nxt in zip(stages, stages[1:]):
        util = DEFAULT_BW_UTIL.get(prev.name, 0.9)
        headroom_gbps = hw.dram_bw_gbps * max(0.0, 1.0 - util)
        if headroom_gbps < 1.0:
            logger.info("no headroom during %s; %s loads on demand",
                        prev.name, nxt.name)
            continue
        transfer_ms = nxt.weights_gb / headroom_gbps * 1e3
        actions.append(PrefetchAction(
            weights_of=nxt.name, during=prev.name,
            # Late-as-possible placeholder; refined with measured stage
            # durations in M3 (e.g. flow weights start at decode step 15).
            start_frac=0.8,
            transfer_ms=transfer_ms,
        ))
        logger.info("prefetch %s (%.2f GB) during %s: %.1f ms at %.0f GB/s headroom",
                    nxt.name, nxt.weights_gb, prev.name, transfer_ms, headroom_gbps)
    return actions
