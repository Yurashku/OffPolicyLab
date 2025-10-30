"""Backend adapters for running Policyscope on alternative data engines."""

from __future__ import annotations

from .pyspark_backend import SparkFrameAdapter

__all__ = ["SparkFrameAdapter"]
