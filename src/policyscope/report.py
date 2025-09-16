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
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .policies import BasePolicy

__all__ = ["decision_summary", "dump_json", "analyze_logs"]


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


def analyze_logs(df: pd.DataFrame, policyB: Optional[BasePolicy] = None) -> str:
    """Проверяет наличие ключевых столбцов в логах и формирует краткий отчёт."""

    lines = ["Проверка входных данных для off-policy оценки:"]

    # базовые колонки
    missing_basic = [c for c in ("user_id", "a_A") if c not in df.columns]
    if missing_basic:
        lines.append(f"- Отсутствуют столбцы {missing_basic}.")
    else:
        lines.append("- Колонки user_id и a_A найдены.")
    if not any(c in df.columns for c in ("accept", "cltv")):
        lines.append("- Отклики (accept/cltv) не найдены.")

    # Replay
    if policyB is not None:
        try:
            a_B = policyB.action_argmax(df)
            share = float(np.mean(a_B == df.get("a_A", -1)))
            lines.append(
                f"- Replay: политика B совпадает с A в {share * 100:.1f}% случаев."
            )
            if share < 0.1:
                lines[-1] += " Требуется больше пересечений для надёжной оценки."
        except Exception:
            lines.append(
                "- Replay: не удалось вычислить пересечение действий A и B."
            )
    else:
        lines.append("- Replay: политика B не передана.")

    # IPS / SNIPS
    if "propensity_A" in df.columns:
        lines.append("- IPS/SNIPS: колонка propensity_A найдена.")
    else:
        lines.append(
            "- IPS/SNIPS: propensities не найдены."
            " Необходимо добавить колонку с π_A(a|x) или обучить модель пропенсити."
        )

    # DM
    feature_candidates = ["age", "income", "risk", "loyal"]
    feats = [c for c in feature_candidates if c in df.columns]
    if feats:
        lines.append(f"- DM: доступны признаки {feats}, ок.")
    else:
        lines.append("- DM: признаки не найдены.")

    # DR
    if "propensity_A" not in df.columns:
        if feats:
            lines.append(
                "- DR: пропенсити отсутствуют, но можно применить DM;"
                " метод DR будет смещён, если модель неточна."
            )
        else:
            lines.append("- DR: нет пропенсити и признаков, метод неприменим.")
    else:
        lines.append("- DR: можно применить, пропенсити присутствуют.")

    for line in lines:
        logging.info(line)
    return "\n".join(lines)
