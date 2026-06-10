"""UMIC: Unified-Memory Inference Compiler.

A thin, measurement-guided AOT compile layer for multi-stage transformer
models on unified-memory edge GPUs (Jetson AGX Thor, SM 11.0).

Design doc: docs/2606_1주차/260610_01_UMIC_iGPU_전용_컴파일엔진_설계서.md
"""

from umic.pipeline import Pipeline, Repeat, Stage
from umic.hw import HardwareProfile

__all__ = ["Pipeline", "Repeat", "Stage", "HardwareProfile"]
__version__ = "0.0.1"
