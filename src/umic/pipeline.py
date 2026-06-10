"""L2: Pipeline IR — the stage-level view no single-graph compiler has.

A model is declared as an ordered list of stages. Each stage owns a weight
footprint and an interface; repeat constructs express autoregressive decode
(until-EOS) and fixed-count ODE loops. The cross-stage prefetch scheduler
and per-stage execution-mode decisions live at this level.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Callable

from umic.hw import THOR, HardwareProfile

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Stage:
    """One pipeline stage backed by an unmodified nn.Module (or callable).

    Attributes:
        name: Unique stage name ("ve", "prefill", "decode", "flow", ...).
        module: The PyTorch module/callable executed by this stage.
        weights_gb: Parameter footprint in GB (drives prefetch scheduling).
        mode: Optional hint ("prefill" / "decode") for capture and caching.
        capturable: Whether the stage is CUDA-Graph safe. Stages with
            dynamic control flow / boolean indexing (e.g. Qwen3VL deepstack)
            must set False and run in stream mode.
    """

    name: str
    module: Any
    weights_gb: float
    mode: str | None = None
    capturable: bool = True


@dataclasses.dataclass
class Repeat:
    """Repeat wrapper: fixed count (`times`) or data-dependent (`until`).

    Attributes:
        stage: The repeated stage.
        times: Fixed iteration count (e.g. 65 ODE steps), or None.
        until: Termination condition name (e.g. "eos"), or None.
        max_iter: Upper bound when `until` is data-dependent.
    """

    stage: Stage
    times: int | None = None
    until: str | None = None
    max_iter: int | None = None

    def __post_init__(self) -> None:
        if (self.times is None) == (self.until is None):
            raise ValueError("Repeat needs exactly one of `times` / `until`")


class Pipeline:
    """Top-level entry point: declare stages, compile, run.

    Example:
        pipe = Pipeline("alpamayo15", stages=[...], hw=THOR)
        engine = pipe.compile()
    """

    def __init__(
        self,
        name: str,
        stages: list[Stage | Repeat],
        hw: HardwareProfile | str | Path = THOR,
        profile_dir: str | Path | None = None,
    ) -> None:
        self.name = name
        self.stages = stages
        self.hw = hw if isinstance(hw, HardwareProfile) else HardwareProfile.from_yaml(hw)
        self.profile_dir = Path(profile_dir) if profile_dir else None

    def flat_stages(self) -> list[Stage]:
        """Return stages in execution order, unwrapping Repeat."""
        return [s.stage if isinstance(s, Repeat) else s for s in self.stages]

    def compile(self) -> Callable[..., Any]:
        """AOT compile: capture → analyze → fuse → memplan → schedule.

        M0 status: capture + analyze are wired; fuse/memplan/runtime are
        progressively enabled per milestone. Anything not yet compiled
        falls back to eager execution (correctness is never blocked).
        """
        from umic.capture import capture_stage
        from umic.schedule import plan_cross_stage_prefetch

        captured = {}
        for stage in self.flat_stages():
            try:
                captured[stage.name] = capture_stage(stage)
                logger.info("captured stage %s", stage.name)
            except Exception as exc:  # noqa: BLE001 — fallback is the design
                logger.warning("stage %s falls back to eager: %s", stage.name, exc)

        schedule = plan_cross_stage_prefetch(self, self.hw)
        logger.info("prefetch schedule: %s", schedule)

        # M0: return an eager executor that simply chains stages.
        # M1+: replace per-stage callables with fused/graph-captured versions.
        def _run(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                "Runtime executor lands in M1; use captured graphs for analysis."
            )

        _run.captured = captured  # type: ignore[attr-defined]
        _run.schedule = schedule  # type: ignore[attr-defined]
        return _run
