"""
Policyscope package.

Предоставляет инструменты для оффлайн‑оценки рекомендационных моделей,
включая генерацию синтетических данных, определение политик и реализацию
оценщиков (IPS, SNIPS, DM, DR, SNDR, Switch-DR) с доверительными интервалами.

Версия пакета определяется переменной `__version__`.
"""

import logging

from policyscope.comparison import (
    RECOMMENDED_CROSSFIT_ESTIMATORS,
    RECOMMENDED_ESTIMATOR,
    RECOMMENDED_PROPENSITY_SOURCE_FALLBACK,
    RECOMMENDED_PROPENSITY_SOURCE_WITH_LOGGED,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

__all__ = [
    "__version__",
    "RECOMMENDED_ESTIMATOR",
    "RECOMMENDED_PROPENSITY_SOURCE_WITH_LOGGED",
    "RECOMMENDED_PROPENSITY_SOURCE_FALLBACK",
    "RECOMMENDED_CROSSFIT_ESTIMATORS",
]

__version__ = "0.1.0"
