"""Nuisance-модели: пропенсити ``pi_A(a|x)`` и модель исхода ``mu(x, a)``.

Обе модели **не зависят от оцениваемой политики B**, поэтому обучаются один раз и
переиспользуются для любого числа кандидатов.

Обе всегда считаются **out-of-fold** (cross-fitting): прогноз для строки делает
модель, которая эту строку не видела. In-sample прогнозы систематически подстроены
под шум данных: гибкая модель пропенсити «запоминает» залогированные действия и
сжимает веса, а модель исхода занижает остатки, на которых держится DR.

Оценённые пропенсити дополнительно ограничиваются снизу **полом для каждого
действия** ``floor_a = propensity_floor * доля действия a в логах``. Без пола
модель, не воспроизводящая ступенчатую логирующую политику (ε-greedy), выдаёт
пропенсити порядка 1e-4 там, где на деле 0.05, и единичные веса в тысячи определяют
всю оценку.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import GammaRegressor, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from policyscope.data import LogData

__all__ = [
    "NuisanceConfig",
    "Nuisance",
    "TwoPartRegressor",
    "make_folds",
    "fit_nuisance",
    "outcome_design",
    "outcome_kind",
]

#: Предел размера дизайн-матрицы модели исхода (строк × столбцов, ~240 МБ в float64).
#: Если блок взаимодействий x×a его превышает, модель строится на главных эффектах.
MAX_DESIGN_CELLS = 30_000_000

#: Сетка обратной силы L2-штрафа (``C``) модели пропенсити по умолчанию. Перебирается от
#: слабой регуляризации к сильной; выбирается значение с наименьшей out-of-fold log-loss.
#: Без подбора (C = 1) модель на многих разреженных признаках переобучается: на Open Bandit
#: Dataset (равномерная политика, 34 действия, 66 признаков) она выдавала пропенсити от 0 до
#: 0.98 вместо 1/34, и DR занижал V(B) на четверть.
PROPENSITY_C_GRID = (10.0, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)

#: До этой ширины дизайна бинарные логистические регрессии решаются методом Ньютона:
#: на узких матрицах он в разы быстрее lbfgs и сходится точнее; на широких его
#: память и время растут как квадрат числа столбцов.
NEWTON_MAX_COLUMNS = 300


@dataclass(frozen=True)
class NuisanceConfig:
    """Настройки nuisance-моделей.

    Attributes:
        propensity_model: sklearn-классификатор с ``predict_proba`` для ``pi_A(a|x)``.
            ``None`` — мультиномиальная логистическая регрессия.
        outcome_model: sklearn-модель для ``mu(x, a)`` или словарь «метрика -> модель»;
            получает признаки, one-hot действия и их попарные произведения. ``None`` (или
            метрика, отсутствующая в словаре) — модель по умолчанию, см. :func:`outcome_kind`.
        n_folds: Число фолдов cross-fitting.
        propensity_floor: Доля от частоты действия, ниже которой оценённое пропенсити
            не опускается. Применяется только к оценённым пропенсити.
        propensity_C: ``C`` логистической модели пропенсити по умолчанию.
        propensity_kind: Класс модели пропенсити по умолчанию: ``"logistic"`` или
            ``"boosting"``. ``None`` — выбрать по out-of-fold log-loss: логистическая регрессия
            с ``C`` из сетки :data:`PROPENSITY_C_GRID` против градиентного бустинга. Выбор
            делается на полной выборке и передаётся bootstrap-репликам.
        propensity_iterations: Число итераций бустинга пропенсити. ``None`` — выбрать ранней
            остановкой (медиана по фолдам лучшей итерации); репликам передаётся готовым.
        interactions: Метрика -> использовать ли блок x×a в модели исхода. ``None`` (или
            метрика, отсутствующая в словаре) — использовать, если дизайн помещается в
            :data:`MAX_DESIGN_CELLS`. Решение принимается на полной выборке, и
            bootstrap-реплики получают готовое: модель не меняется от реплики к реплике.
    """

    propensity_model: Any = None
    outcome_model: Any = None
    n_folds: int = 5
    propensity_floor: float = 0.1
    propensity_C: Optional[float] = None
    propensity_kind: Optional[str] = None
    propensity_iterations: Optional[int] = None
    interactions: Optional[Mapping[str, bool]] = None

    def model_for(self, metric: str) -> Any:
        """Пользовательская модель исхода для метрики или ``None`` (модель по умолчанию)."""
        if isinstance(self.outcome_model, Mapping):
            return self.outcome_model.get(metric)
        return self.outcome_model

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("n_folds должен быть не меньше 2")
        if not 0.0 < self.propensity_floor < 1.0:
            raise ValueError("propensity_floor должен лежать в (0, 1)")
        if self.propensity_kind not in (None, "logistic", "boosting"):
            raise ValueError("propensity_kind: None, 'logistic' или 'boosting'")
        if self.propensity_iterations is not None and (
            self.propensity_kind != "boosting" or self.propensity_iterations < 1
        ):
            raise ValueError("propensity_iterations — целое ≥ 1 и только при propensity_kind='boosting'")
        if self.propensity_C is not None and (self.propensity_kind == "boosting" or self.propensity_C <= 0):
            raise ValueError("propensity_C — положительное число и только для логистической модели")


@dataclass(frozen=True)
class Nuisance:
    """Out-of-fold прогнозы nuisance-моделей, выровненные по строкам выборки.

    Attributes:
        p_taken: Пропенсити залогированного действия, используемые в весах
            (для оценённых — с полом).
        source: ``"logged"`` или ``"estimated"``.
        p_raw: Оценённое пропенсити без пола (только для оценённых).
        p_max: Максимальная оценённая вероятность по действиям — признак
            детерминированности A (только для оценённых).
        floor: Пол для каждого действия ``(k,)`` (только для оценённых).
        unseen_share: Доля строк, чьё действие не встретилось в обучающем фолде.
        mu: Метрика -> прогнозы ``(n, k)``.
        fit_seconds: Время обучения без подбора ``C`` (для прогноза длительности bootstrap).
        interactions: Метрика -> использован ли блок x×a в модели исхода.
        design_notes: Метрика -> дизайн модели исхода и причина (для отчёта).
        propensity_C: ``C`` логистической модели пропенсити по умолчанию (``None`` — выбран
            бустинг, пропенсити залогированы или модель задана пользователем).
        propensity_kind: ``"logistic"`` / ``"boosting"`` для модели по умолчанию, иначе ``None``.
        propensity_iterations: Число итераций бустинга, если выбран бустинг: медиана по фолдам
            лучшей итерации ранней остановки (см. :func:`_early_stopping_iterations`).
        propensity_losses: Out-of-fold log-loss кандидатов, если модель выбиралась: ключ — ``C``
            логистической регрессии или ``"boosting"``.
        propensity_note: Почему бустинг не участвовал в выборе (например, слишком редкое
            действие для ранней остановки); ``None`` — участвовал или выбор не делался.
        p_matrix: Оценённые out-of-fold вероятности всех действий ``(n, k)`` без пола
            (только для оценённых) — для проверки калибровки на решениях кандидата.
    """

    p_taken: np.ndarray
    source: str
    p_raw: Optional[np.ndarray]
    p_max: Optional[np.ndarray]
    floor: Optional[np.ndarray]
    unseen_share: float
    mu: dict[str, np.ndarray]
    fit_seconds: float
    interactions: dict[str, bool] = field(default_factory=dict)
    design_notes: dict[str, str] = field(default_factory=dict)
    propensity_C: Optional[float] = None
    propensity_kind: Optional[str] = None
    propensity_iterations: Optional[int] = None
    propensity_losses: dict[Any, float] = field(default_factory=dict)
    p_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    propensity_note: Optional[str] = None

    def floored(self, a: np.ndarray, factor: float) -> np.ndarray:
        """Пропенсити с полом, умноженным на ``factor`` (анализ чувствительности)."""
        if self.source != "estimated":
            return self.p_taken
        assert self.p_raw is not None and self.floor is not None
        return np.maximum(self.p_raw, factor * self.floor[a])


def _default_propensity_model(C: float = 1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))


def _boosting_propensity_model(iterations: Optional[int] = None):
    """Бустинг для пропенсити: описывает ступенчатые правила (argmax скоринга, смеси версий).

    Без ``iterations`` — ранняя остановка по отложенным 10% переданных строк: обучение
    прекращается, когда log-loss на них 10 итераций подряд не улучшается. Так выбирается число
    итераций (:func:`_early_stopping_iterations`); OOF-прогнозы и bootstrap-реплики затем
    обучают бустинг ровно с этим числом итераций, без отложенной части. L2-штраф ограничивает
    шаг Ньютона в листе с очень редким действием: без него одна строка такого действия давала
    остальным строкам листа вероятность этого действия около 1, и бустинг проигрывал выбор
    модели. Зерно фиксировано — результат детерминирован.
    """
    if iterations is not None:
        return HistGradientBoostingClassifier(
            max_iter=iterations, early_stopping=False, l2_regularization=1.0, random_state=0,
        )
    return HistGradientBoostingClassifier(
        max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
        l2_regularization=1.0, random_state=0,
    )


class TwoPartRegressor(RegressorMixin, BaseEstimator):
    """Модель неотрицательной награды с нулями: ``E[r|z] = P(r > 0|z) · E[r | r > 0, z]``.

    Первая часть — логистическая регрессия, вторая — Gamma-GLM с log-связью на строках с
    ``r > 0``. Так устроены выручка, сумма покупки, LTV: чаще всего ноль, иначе
    положительная величина с длинным хвостом. Линейная модель на таких данных выдаёт
    отрицательные прогнозы, а их обрезка смещает оценку DM на величину порядка его
    стандартной ошибки — это показали симуляции (см. docs/estimators.md).

    Прогноз положительной части не превышает наблюдённый максимум ``r``. Log-связь
    экстраполирует экспоненциально, и в областях без данных (слабое перекрытие) прогноз
    без ограничения уходит в разы выше любого наблюдения: в симуляциях это смещало и DM,
    и DR. Ограничение действует только там, где модель экстраполирует.

    Args:
        C: Обратная сила L2-штрафа логистической части (как в ``LogisticRegression``).
            Штраф Gamma-части согласован с ним: ``alpha = 1 / (C · n_pos)``.
    """

    def __init__(self, C: float = 1.0) -> None:
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if (y < 0).any():
            raise ValueError("TwoPartRegressor ожидает неотрицательную награду")
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        positive = y > 0
        n_pos = int(positive.sum())
        self.p_const_ = float(positive.mean())
        self.classifier_ = None
        if 0 < n_pos < len(y):
            self.classifier_ = LogisticRegression(
                C=self.C, max_iter=1000, solver=_binary_solver(Z.shape[1]),
            ).fit(Z, positive)
        self.m_const_ = float(y[positive].mean()) if n_pos else 0.0
        self.cap_ = float(y.max())
        self.regressor_ = None
        if n_pos >= 10:
            self.regressor_ = GammaRegressor(
                alpha=1.0 / (self.C * n_pos), solver="newton-cholesky", max_iter=300,
            ).fit(Z[positive], y[positive])
        return self

    def predict(self, X):
        Z = self.scaler_.transform(np.asarray(X, dtype=float))
        p = (self.classifier_.predict_proba(Z)[:, 1] if self.classifier_ is not None
             else np.full(len(Z), self.p_const_))
        m = self.regressor_.predict(Z) if self.regressor_ is not None else np.full(len(Z), self.m_const_)
        return p * np.minimum(m, self.cap_)


def outcome_kind(r: np.ndarray, binary: bool) -> str:
    """Тип модели исхода по умолчанию для вектора наград.

    - ``binary`` — логистическая регрессия;
    - ``two_part`` — неотрицательная награда, где заметная доля нулей (выручка):
      :class:`TwoPartRegressor`;
    - ``linear`` — иначе: Ridge.
    """
    if binary:
        return "binary"
    if (r >= 0).all() and np.mean(r == 0) >= 0.05 and np.sum(r > 0) >= 20:
        return "two_part"
    return "linear"


OUTCOME_KIND_LABELS = {
    "binary": "LogisticRegression",
    "two_part": "двухчастная: LogisticRegression × GammaRegressor (log)",
    "linear": "Ridge",
}


def _binary_solver(n_columns: int) -> str:
    return "newton-cholesky" if n_columns <= NEWTON_MAX_COLUMNS else "lbfgs"


def _default_outcome_model(kind: str, n_columns: int):
    if kind == "binary":
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, solver=_binary_solver(n_columns)),
        )
    if kind == "two_part":
        return TwoPartRegressor()
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def make_folds(groups: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Номер фолда для каждой строки; все строки одной группы — в одном фолде.

    Группа — кластер (клиент) или, в bootstrap-реплике, исходная строка: копии одной
    строки не должны попадать одновременно в обучение и в прогноз.
    """
    unique_groups, group_of_row = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    if n_groups < n_folds:
        raise ValueError(f"Групп ({n_groups}) меньше, чем фолдов ({n_folds})")
    fold_of_group = rng.permutation(n_groups) % n_folds
    return fold_of_group[group_of_row]


def outcome_design(X: np.ndarray, a: np.ndarray, k: int, interactions: bool = True) -> np.ndarray:
    """Дизайн-матрица модели исхода: ``[x, onehot(a), x * onehot(a)]``.

    Блок произведений даёт каждому действию собственные наклоны: без него модель
    отличает действия только сдвигом и не может выразить, что оффер работает по-разному
    для разных клиентов. При ``interactions=False`` — только ``[x, onehot(a)]``.
    """
    onehot = np.zeros((len(a), k))
    onehot[np.arange(len(a)), a] = 1.0
    if not interactions:
        return np.hstack([X, onehot])
    inter = (X[:, :, None] * onehot[:, None, :]).reshape(len(a), -1)
    return np.hstack([X, onehot, inter])


def _design_width(p: int, k: int, interactions: bool) -> int:
    return p + k + (p * k if interactions else 0)


def _propensity_oof(
    X: np.ndarray, a: np.ndarray, k: int, folds: np.ndarray, n_folds: int, model: Any,
    fitted_models: Optional[list] = None,
) -> tuple[np.ndarray, int]:
    probs = np.zeros((len(a), k))
    unseen = 0
    for f in range(n_folds):
        train, hold = folds != f, folds == f
        classes = np.unique(a[train])
        if len(classes) == 1:
            probs[np.ix_(hold, classes)] = 1.0
        else:
            fitted = clone(model).fit(X[train], a[train])
            probs[np.ix_(hold, np.asarray(fitted.classes_, dtype=int))] = fitted.predict_proba(X[hold])
            if fitted_models is not None:
                fitted_models.append(fitted)
        unseen += int(np.sum(~np.isin(a[hold], classes)))
    return probs, unseen


def _early_stopping_iterations(
    X: np.ndarray, a: np.ndarray, folds: np.ndarray, n_folds: int,
) -> tuple[Optional[int], Optional[str]]:
    """Число итераций бустинга: медиана по фолдам лучшей итерации ранней остановки.

    На обучающей части каждого фолда бустинг обучается с ранней остановкой, и берётся итерация
    с лучшим log-loss на отложенных строках. Это не ``n_iter_``: обучение останавливается на
    10 итераций позже лучшей (терпение ранней остановки), а лишние итерации переобучают модель
    и раздувают веса. Медиана по фолдам меньше зависит от случайного разбиения, чем одна
    остановка на полной выборке.

    Действия с одной строкой в фолде в этой подгонке не участвуют: стратифицированной
    отложенной части нужны две строки каждого класса, а число итераций — лишь гиперпараметр.

    Returns:
        Число итераций (``None``, если ранняя остановка невозможна ни в одном фолде) и причина.
    """
    best: list[int] = []
    for f in range(n_folds):
        train = folds != f
        X_train, a_train = X[train], a[train]
        labels, counts = np.unique(a_train, return_counts=True)
        keep = np.isin(a_train, labels[counts >= 2])
        if len(np.unique(a_train[keep])) < 2:
            continue
        try:
            model = _boosting_propensity_model().fit(X_train[keep], a_train[keep])
        except ValueError:  # отложенной части не хватает строк на все действия
            continue
        best.append(int(np.argmax(model.validation_score_)))
    if not best:
        return None, "бустинг не сравнивался: для ранней остановки слишком мало строк редких действий"
    return max(int(np.median(best)), 1), None


def _tune_propensity(
    X: np.ndarray, a: np.ndarray, k: int, folds: np.ndarray, n_folds: int, logistic_only: bool = False,
) -> tuple[str, Optional[float], Optional[int], np.ndarray, int, dict[Any, float], float, Optional[str]]:
    """Выбирает модель пропенсити по умолчанию по out-of-fold log-loss.

    Кандидаты — логистическая регрессия (``C`` подбирается по сетке) и градиентный
    бустинг. Логистическая верна, когда A — softmax по линейным скорам; бустинг нужен для
    ступенчатых правил: argmax скоринга, ε-greedy, смесь версий модели. Логистическая
    регрессия такие правила сглаживает и приписывает вероятность действиям, которые A этим
    клиентам не показывала, — ложное перекрытие, которое не видно по среднему весу.

    Log-loss — правдоподобие залогированных действий, то есть ровно то, что модель
    пропенсити должна предсказывать (в отличие от модели исхода, где ошибка на
    залогированных действиях не видит ошибок в контрастах).

    Сетка идёт от слабой регуляризации к сильной и обрывается после двух значений подряд,
    не улучшивших лучший результат. Направление важно: при сильной регуляризации все модели почти
    маргинальные, потери там лежат на плато, и шум на плато выглядел бы как рост — путь
    от сильной регуляризации останавливался бы, не дойдя до моделей, которые учат
    контекст. Выбор — одно значение из сетки, поэтому OOF-прогнозы победителя
    используются как есть.

    Число итераций бустинга выбирается ранней остановкой (:func:`_early_stopping_iterations`),
    а его OOF-прогнозы и log-loss считаются бустингом с этим числом итераций — ровно той
    процедурой, которую повторяют bootstrap-реплики. Иначе точечная оценка и реплики были бы
    разными оценщиками, и интервалы и p-value смешивали бы их. Если ранняя остановка
    невозможна (слишком редкие действия), бустинг не участвует, и причина попадает в отчёт.

    ``logistic_only`` — подобрать только ``C`` логистической регрессии.

    Последний элемент — время на отвергнутых кандидатов: bootstrap-реплики их не
    обучают, и прогноз длительности bootstrap не должен их учитывать.
    """
    rows = np.arange(len(a))
    losses: dict[float, float] = {}
    best: Optional[tuple[float, float, np.ndarray, int, float]] = None
    worse, total = 0, 0.0
    for C in PROPENSITY_C_GRID:
        start = time.perf_counter()
        probs, unseen = _propensity_oof(X, a, k, folds, n_folds, _default_propensity_model(C))
        seconds = time.perf_counter() - start
        total += seconds
        loss = float(-np.mean(np.log(np.clip(probs[rows, a], 1e-12, None))))
        losses[C] = loss
        if best is None or loss < best[1]:
            best, worse = (C, loss, probs, unseen, seconds), 0
        else:
            worse += 1
            if worse == 2:
                break
    assert best is not None
    if logistic_only or len(np.unique(a)) < 2:
        return "logistic", best[0], None, best[2], best[3], losses, total - best[4], None
    start = time.perf_counter()
    iterations, note = _early_stopping_iterations(X, a, folds, n_folds)
    tuning = time.perf_counter() - start
    total += tuning
    if iterations is None:
        return "logistic", best[0], None, best[2], best[3], losses, total - best[4], note
    start = time.perf_counter()
    probs, unseen = _propensity_oof(X, a, k, folds, n_folds, _boosting_propensity_model(iterations))
    seconds = time.perf_counter() - start
    total += seconds
    losses["boosting"] = float(-np.mean(np.log(np.clip(probs[rows, a], 1e-12, None))))
    if losses["boosting"] < best[1]:
        return "boosting", None, iterations, probs, unseen, losses, total - seconds, None
    return "logistic", best[0], None, best[2], best[3], losses, total - best[4], None


def _outcome_oof(
    X: np.ndarray, a: np.ndarray, r: np.ndarray, k: int, binary: bool,
    folds: np.ndarray, n_folds: int, model: Any, kind: str, interactions: bool = True,
) -> np.ndarray:
    mu = np.zeros((len(a), k))
    width = _design_width(X.shape[1], k, interactions)
    base = model if model is not None else _default_outcome_model(kind, width)
    for f in range(n_folds):
        train, hold = folds != f, folds == f
        n_hold = int(hold.sum())
        r_train = r[train]
        if np.unique(r_train).size < 2:
            mu[hold] = float(r_train.mean())
            continue
        fitted = clone(base).fit(outcome_design(X[train], a[train], k, interactions), r_train)
        # По одному действию: матрица для всех k сразу в k раз больше обучающей.
        for j in range(k):
            design = outcome_design(X[hold], np.full(n_hold, j), k, interactions)
            if binary and hasattr(fitted, "predict_proba"):
                pred = fitted.predict_proba(design)[:, 1]
            else:
                pred = fitted.predict(design)
                if binary:  # регрессор на бинарной награде: прогноз — вероятность
                    pred = np.clip(pred, 0.0, 1.0)
            # Непрерывные прогнозы не обрезаются: обрезка по [min r, max r] смещает DM.
            mu[hold, j] = np.asarray(pred, dtype=float)
    if not np.isfinite(mu).all():
        raise ValueError(
            "Модель исхода выдала нечисловые прогнозы (inf/NaN). Проверьте признаки на выбросы "
            "или задайте более устойчивую outcome_model."
        )
    return mu


def _outcome_with_design(
    X: np.ndarray, a: np.ndarray, r: np.ndarray, k: int, binary: bool,
    folds: np.ndarray, n_folds: int, model: Any, kind: str, interactions: Optional[bool],
) -> tuple[np.ndarray, bool, str]:
    """OOF-прогнозы модели исхода и её дизайн (с блоком x×a или без).

    Блок x×a нужен всегда, когда он помещается в память: оценка политики держится на
    контрастах ``mu(x, a) − mu(x, a')``, а их неоднородность по клиентам без него не
    выразить. Выбор по out-of-fold ошибке здесь не годится: она считается на
    залогированных действиях и почти не видит ошибок в контрастах (на Hillstrom такой
    выбор отбрасывал x×a, и DM терял весь эффект). Поэтому x×a отключается только по
    размеру матрицы — тогда модель отличает действия лишь сдвигом, и это видно в отчёте.
    """
    if interactions is None:
        width = _design_width(X.shape[1], k, True)
        interactions = model is not None or len(a) * width <= MAX_DESIGN_CELLS
        if interactions:
            note = "с x×a"
        else:
            size = f"{len(a):,} × {width:,}".replace(",", " ")
            note = f"без x×a: матрица с ними слишком велика ({size} ячеек)"
    else:
        note = "с x×a" if interactions else "без x×a"
    mu = _outcome_oof(X, a, r, k, binary, folds, n_folds, model, kind, interactions)
    return mu, interactions, note


def fit_nuisance(
    data: LogData,
    config: NuisanceConfig,
    *,
    seed: int,
    positions: Optional[np.ndarray] = None,
) -> Nuisance:
    """Обучает nuisance-модели out-of-fold на всей выборке или на bootstrap-реплике.

    Args:
        data: Подготовленные логи.
        config: Настройки моделей.
        seed: Зерно разбиения на фолды.
        positions: Позиции строк реплики; ``None`` — вся выборка.
    """
    # Модели обучаются в одном потоке (BLAS и OpenMP). Многопоточный BLAS на небольших
    # матрицах в разы медленнее однопоточного, а OpenMP-бустинг на части машин (например,
    # в WSL) медленнее в десятки раз: синхронизация потоков дороже самих вычислений.
    # Параллелизм у bootstrap и так есть — на уровне реплик.
    with threadpool_limits(limits=1):
        return _fit_nuisance(data, config, seed=seed, positions=positions)


def _fit_nuisance(
    data: LogData, config: NuisanceConfig, *, seed: int, positions: Optional[np.ndarray]
) -> Nuisance:
    if isinstance(config.outcome_model, Mapping):
        extra = sorted(set(config.outcome_model) - set(data.reward_cols))
        if extra:
            raise ValueError(f"outcome_model задан для метрик, которых нет в reward: {extra}")
    start = time.perf_counter()
    idx = np.arange(data.n) if positions is None else np.asarray(positions)
    X = data.X[idx]
    if X.shape[1] == 0:  # оценка без контекста
        X = np.zeros((len(idx), 1))
    a = data.a[idx]
    k = data.k
    groups = data.clusters[idx] if data.is_clustered else idx
    folds = make_folds(groups, config.n_folds, np.random.default_rng(seed))

    unseen = 0
    propensity_C: Optional[float] = None
    propensity_kind: Optional[str] = None
    propensity_iterations: Optional[int] = None
    losses: dict[Any, float] = {}
    tuning_seconds = 0.0
    propensity_note: Optional[str] = None
    probs = None
    if data.propensity is not None:
        p_taken = data.propensity[idx]
        source, p_raw, p_max, floor = "logged", None, None, None
    else:
        if config.propensity_model is not None:
            probs, unseen = _propensity_oof(X, a, k, folds, config.n_folds, config.propensity_model)
        elif config.propensity_kind == "boosting":
            propensity_kind, propensity_iterations = "boosting", config.propensity_iterations
            if propensity_iterations is None:
                propensity_iterations, reason = _early_stopping_iterations(X, a, folds, config.n_folds)
                if propensity_iterations is None:
                    raise ValueError(f"{reason}. Задайте propensity_iterations явно.")
            probs, unseen = _propensity_oof(
                X, a, k, folds, config.n_folds, _boosting_propensity_model(propensity_iterations),
            )
        elif config.propensity_C is not None:
            propensity_kind, propensity_C = "logistic", config.propensity_C
            probs, unseen = _propensity_oof(
                X, a, k, folds, config.n_folds, _default_propensity_model(propensity_C),
            )
        else:
            (propensity_kind, propensity_C, propensity_iterations, probs, unseen, losses,
             tuning_seconds, propensity_note) = _tune_propensity(
                X, a, k, folds, config.n_folds, logistic_only=config.propensity_kind == "logistic",
            )
        rows = np.arange(len(a))
        p_raw = probs[rows, a]
        p_max = probs.max(axis=1)
        floor = config.propensity_floor * np.bincount(a, minlength=k) / len(a)
        p_taken = np.maximum(p_raw, floor[a])
        source = "estimated"

    mu: dict[str, np.ndarray] = {}
    interactions: dict[str, bool] = {}
    notes: dict[str, str] = {}
    for metric in data.reward_cols:
        mu[metric], interactions[metric], notes[metric] = _outcome_with_design(
            X, a, data.rewards[metric][idx], k, data.binary[metric],
            folds, config.n_folds, config.model_for(metric),
            outcome_kind(data.rewards[metric], data.binary[metric]),
            (config.interactions or {}).get(metric),
        )
    return Nuisance(
        p_taken=p_taken,
        source=source,
        p_raw=p_raw,
        p_max=p_max,
        floor=floor,
        unseen_share=unseen / len(a),
        mu=mu,
        fit_seconds=time.perf_counter() - start - tuning_seconds,
        interactions=interactions,
        design_notes=notes,
        propensity_C=propensity_C,
        propensity_kind=propensity_kind,
        propensity_iterations=propensity_iterations,
        propensity_losses=losses,
        p_matrix=probs,
        propensity_note=propensity_note,
    )
