"""
Inference and evaluation.

The metric registry is re-exported here so evaluation code can depend on
`inference` alone. It is deliberately a registry rather than a fixed list:
pixel similarity metrics were shown to be poor indicators of garment
correctness for this task, so saree-specific measures can be added without
touching the evaluation driver.
"""

from inference.metrics import Metric, available, build, describe, register
from inference.pipeline import PipelineOutput, TryOnPipeline

__all__ = [
    "TryOnPipeline",
    "PipelineOutput",
    "Metric",
    "available",
    "build",
    "describe",
    "register",
]
