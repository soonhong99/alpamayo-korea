"""L1 frontend: capture a stage's ATen graph with torch.fx / torch.export.

Capture is per-stage (never whole-model): a failure in one stage must not
take down the others. Anything uncapturable stays eager.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def capture_stage(stage: Any, example_inputs: tuple | None = None) -> torch.fx.GraphModule:
    """Capture one stage as an fx GraphModule.

    Tries torch.export first (ATen-level, decompositions applied), then
    falls back to symbolic_trace (python-level). Raises on total failure —
    the caller treats that stage as eager.

    Args:
        stage: A Stage (or any object with .module).
        example_inputs: Concrete example inputs required by torch.export.

    Returns:
        The captured GraphModule.
    """
    module = stage.module if hasattr(stage, "module") else stage

    if example_inputs is not None:
        try:
            exported = torch.export.export(module, example_inputs)
            return exported.module()
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch.export failed (%s); trying symbolic_trace", exc)

    return torch.fx.symbolic_trace(module)
