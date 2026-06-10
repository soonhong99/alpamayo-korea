"""Hardware profile: the resource model the scheduler plans against.

On an iGPU every resource shares one DRAM bus, so bandwidth is treated as
a global budget rather than a per-device property.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class HardwareProfile:
    """Static description of the target unified-memory SoC.

    Attributes:
        name: Human-readable device name.
        sm_arch: Compute capability string, e.g. "11.0".
        dram_bw_gbps: Peak one-directional DRAM read bandwidth (GB/s).
        l2_bytes: GPU L2 cache size in bytes.
        l2_persist_max_bytes: Max bytes pinnable via cudaAccessPolicyWindow.
        cpu_cores: Number of CPU cores sharing the DRAM bus.
        kernel_launch_us: Measured per-kernel launch overhead (microseconds).
    """

    name: str
    sm_arch: str
    dram_bw_gbps: float
    l2_bytes: int
    l2_persist_max_bytes: int
    cpu_cores: int
    kernel_launch_us: float = 3.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HardwareProfile":
        """Load a profile from a YAML file (see configs/hw/thor.yaml)."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        logger.info("Loaded hardware profile: %s", raw.get("name"))
        return cls(**raw)


# Built-in profile for Jetson AGX Thor (values confirmed by measurement,
# see docs/2606_1주차/260608_01 and memory project_thor_cache_specs).
THOR = HardwareProfile(
    name="Jetson AGX Thor",
    sm_arch="11.0",
    dram_bw_gbps=231.0,
    l2_bytes=32 * 1024 * 1024,
    l2_persist_max_bytes=24 * 1024 * 1024,  # leave headroom; tune on device
    cpu_cores=12,
)
