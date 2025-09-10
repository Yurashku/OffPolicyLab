"""
Policyscope package.

Предоставляет инструменты для оффлайн‑оценки рекомендационных моделей,
включая генерацию синтетических данных, определение политик и реализацию
оценщиков (IPS, SNIPS, DM, DR, SNDR, Switch-DR) с доверительными интервалами.

Версия пакета определяется переменной `__version__`.
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

__all__ = ["__version__"]

__version__ = "0.1.0"
