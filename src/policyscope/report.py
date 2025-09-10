"""
policyscope.report
===================

Функции для формирования текстового отчёта по результатам off‑policy оценки
и для записи JSON‑файлов. В отчёте указывается значение политики A,
значение политики B, оценка ATE (разности) и доверительные интервалы, а
также делается заключение относительно превосходства одной политики над
другой при заданном бизнес‑пороге.
"""

from __future__ import annotations

import json
from typing import Dict

__all__ = ["decision_summary", "dump_json"]


def decision_summary(res: Dict, metric_name: str, business_threshold: float = 0.0) -> str:
    """Формирует текстовый вывод по результатам бутстрэпа.

    Parameters
    ----------
    res : dict
        Результат функции `paired_bootstrap_ci`.
    metric_name : str
        Название метрики (например, "Отклик" или "CLTV").
    business_threshold : float
        Минимальный прирост, который считается значимым (например, >0.005 для
        увеличения на 0.5 п.п. для отклика). Если границы CI лежат выше этого
        порога, политика B считается лучше.

    Returns
    -------
    str
        Сформированный отчёт.
    """
    V_A = res["V_A"]
    V_B = res["V_B"]
    D = res["Delta"]
    A_lo, A_hi = res["V_A_CI"]
    B_lo, B_hi = res["V_B_CI"]
    D_lo, D_hi = res["Delta_CI"]

    lines = []
    lines.append(f"Метрика: {metric_name}")
    lines.append(f"V(A) = {V_A:.6f} (95% CI: {A_lo:.6f} .. {A_hi:.6f})")
    lines.append(f"V(B) = {V_B:.6f} (95% CI: {B_lo:.6f} .. {B_hi:.6f})")
    lines.append(f"ATE (B−A) = {D:.6f} (95% CI: {D_lo:.6f} .. {D_hi:.6f})")

    if D_lo > business_threshold:
        lines.append(f"Решение: модель B лучше A, поскольку нижняя граница CI превышает порог {business_threshold}.")
    elif D_hi < -business_threshold:
        lines.append(f"Решение: модель A лучше B, поскольку верхняя граница CI ниже -{business_threshold}.")
    else:
        lines.append("Решение: статистически значимого отличия не обнаружено или эффект слишком мал.")
    return "\n".join(lines)


def dump_json(path: str, obj) -> None:
    """Записывает объект в JSON‑файл с красивым форматированием."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
