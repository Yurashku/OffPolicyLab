"""Policyscope — офлайн-оценка политик рекомендаций по логам (off-policy evaluation).

Задача: по логам работающей политики **A** понять, что даст политика-кандидат **B**,
без онлайн A/B-теста — с доверительным интервалом и честным ответом на вопрос,
можно ли этой оценке верить.

Быстрый старт::

    import policyscope as ps

    ev = ps.Evaluator(logs, action="offer", reward=["accepted", "revenue"],
                      propensity="propensity", cluster="client_id")
    report = ev.evaluate(policy_b)   # DataFrame вероятностей: колонки — офферы
    report                           # в Jupyter рендерится отчётом
"""

from __future__ import annotations

import logging

from policyscope import synthetic
from policyscope.diagnostics import Finding, Thresholds
from policyscope.evaluator import Evaluator, evaluate
from policyscope.policy import epsilon_greedy, greedy, softmax
from policyscope.report import Leaderboard, MetricReport, Report

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "Evaluator",
    "evaluate",
    "Report",
    "MetricReport",
    "Leaderboard",
    "Finding",
    "Thresholds",
    "greedy",
    "softmax",
    "epsilon_greedy",
    "synthetic",
]
