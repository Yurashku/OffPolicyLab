"""
policyscope.report
===================

Функции для формирования текстового отчёта по результатам off‑policy оценки
и для записи JSON‑файлов. В отчёте указывается значение политики A,
значение политики B, оценка delta (разности политик) и доверительные интервалы, а
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
    if hasattr(res, "to_dict"):
        res = res.to_dict()

    V_A = res["V_A"]
    V_B = res["V_B"]
    D = res["Delta"]
    alpha = float(res.get("alpha", res.get("inference_alpha", 0.05)))
    ci_level = int(round((1.0 - alpha) * 100))
    a_ci = res.get("V_A_CI")
    b_ci = res.get("V_B_CI")
    d_ci = res.get("Delta_CI")

    lines = []
    lines.append(f"Метрика: {metric_name}")
    if a_ci is not None:
        A_lo, A_hi = a_ci
        lines.append(f"V(A) = {V_A:.6f} ({ci_level}% CI: {A_lo:.6f} .. {A_hi:.6f})")
    else:
        lines.append(f"V(A) = {V_A:.6f} (CI недоступен)")
    if b_ci is not None:
        B_lo, B_hi = b_ci
        lines.append(f"V(B) = {V_B:.6f} ({ci_level}% CI: {B_lo:.6f} .. {B_hi:.6f})")
    else:
        lines.append(f"V(B) = {V_B:.6f} (CI недоступен)")
    if d_ci is not None:
        D_lo, D_hi = d_ci
        lines.append(f"Delta (B−A) = {D:.6f} ({ci_level}% CI: {D_lo:.6f} .. {D_hi:.6f})")
    else:
        lines.append(f"Delta (B−A) = {D:.6f} (CI недоступен)")

    if d_ci is not None:
        if D_lo > business_threshold:
            lines.append(f"Решение: модель B лучше A, поскольку нижняя граница CI превышает порог {business_threshold}.")
        elif D_hi < -business_threshold:
            lines.append(f"Решение: модель A лучше B, поскольку верхняя граница CI ниже -{business_threshold}.")
        else:
            lines.append("Решение: статистически значимого отличия не обнаружено или эффект слишком мал.")
    elif res.get("is_significant") is True:
        lines.append("Решение: обнаружено статистически значимое отличие, но без CI интерпретация менее устойчива.")
    else:
        lines.append("Решение: CI не передан; итог следует трактовать как предварительный.")
    recommendation = res.get("recommendation")
    trust_level = res.get("trust_level")
    if trust_level is not None:
        lines.append(f"Уровень доверия к оценке: {trust_level}.")
    if recommendation:
        lines.append(f"Рекомендация: {recommendation}")
    return "\n".join(lines)


def dump_json(path: str, obj) -> None:
    """Записывает объект в JSON‑файл с красивым форматированием."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def analyze_logs(
    df: pd.DataFrame,
    policyB: Optional[BasePolicy] = None,
    *,
    action_a_col: str = "a_A",
    action_b_col: str = "a_B",
    user_id_col: str = "user_id",
    propensity_col: str = "propensity_A",
    targets: tuple[str, ...] = ("accept", "cltv"),
    feature_cols: Optional[list[str]] = None,
) -> str:
    """Проверяет наличие ключевых столбцов в логах и формирует краткий отчёт."""

    lines = [
        "Проверка входных данных для off-policy оценки:",
        "- Рекомендуемый high-level путь: compare_policies(...) или OPEEvaluator(...).evaluate_summary(...).",
    ]

    # базовые колонки
    missing_basic = [c for c in (user_id_col, action_a_col) if c not in df.columns]
    if missing_basic:
        lines.append(f"- Отсутствуют столбцы {missing_basic}.")
    else:
        lines.append(f"- Колонки {user_id_col} и {action_a_col} найдены.")
    if not any(c in df.columns for c in targets):
        lines.append(f"- Отклики {targets} не найдены.")

    if action_b_col in df.columns:
        share_ab = float(np.mean(df[action_b_col] == df[action_a_col])) if action_a_col in df.columns else 0.0
        lines.append(f"- Найден столбец {action_b_col}; пересечение с {action_a_col}: {share_ab * 100:.1f}%.")

    # Replay
    if policyB is not None:
        try:
            a_B = policyB.action_argmax(df)
            share = float(np.mean(a_B == df.get(action_a_col, -1)))
            lines.append(f"- Replay overlap A/B: {share * 100:.1f}%.")
            if share < 0.1:
                lines.append("- Replay: низкий overlap, оценка может быть шумной и зависимой от support логирующей политики.")
            else:
                lines.append("- Replay: интерпретируйте как диагностический baseline, а не как универсально несмещённую оценку.")
        except Exception:
            lines.append("- Replay: не удалось вычислить пересечение действий A и B.")
    else:
        lines.append("- Replay: политика B не передана; overlap-диагностика недоступна.")

    # Propensity source modes
    if propensity_col in df.columns:
        lines.append(f"- Propensity: колонка {propensity_col} найдена; режим auto сможет использовать logged propensity path.")
        lines.append("- Propensity: estimated path также доступен через propensity_source='estimated'.")
    else:
        lines.append(
            f"- Propensity: колонка {propensity_col} не найдена; режим auto перейдёт в estimated propensity path."
        )
        lines.append("- Propensity: strict logged path требует валидную propensity_col.")

    # Feature availability for DM/DR family
    if feature_cols is None:
        feature_candidates = ["age", "income", "risk", "loyal"]
        feats = [c for c in feature_candidates if c in df.columns]
    else:
        feats = [c for c in feature_cols if c in df.columns]
    if feats:
        lines.append(f"- DM: доступны признаки {feats}, ок.")
    else:
        lines.append("- DM/DR-family: признаки не найдены; модели nuisance могут быть нестабильны.")

    if feats:
        lines.append("- DR/SNDR/Switch-DR: применимы через официальный comparison API; проверяйте CI/p-value и diagnostics вместе.")
        lines.append("- Cross-fit: опционально рекомендуется для дополнительного bias-hardening в DR-family режимах.")

    for line in lines:
        logging.info(line)
    return "\n".join(lines)
