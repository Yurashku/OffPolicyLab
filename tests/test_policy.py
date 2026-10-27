"""Политики: хелперы и приведение любого входа к матрице в порядке действий логов."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import policyscope as ps
from policyscope.data import prepare_logs
from policyscope.policy import policy_matrix

ACTIONS = ["card", "deposit", "loan"]


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    n = 50
    logs = pd.DataFrame({
        "x": rng.normal(size=n),
        "offer": rng.choice(ACTIONS, n),
        "chosen": rng.choice(ACTIONS, n),
        "p": 1 / 3,
        "r": rng.integers(0, 2, n),
    })
    data = prepare_logs(logs, action="offer", reward="r", propensity="p", features=["x"])
    probs = pd.DataFrame(rng.dirichlet(np.ones(3), n), columns=ACTIONS)
    return logs, data, probs


def as_matrix(policy, logs, data):
    return policy_matrix(policy, data, logs.drop(columns=["r", "p"]))


class TestHelpers:
    def test_greedy_splits_ties(self):
        out = ps.greedy(pd.DataFrame([[1.0, 1.0, 0.0]], columns=ACTIONS))
        np.testing.assert_allclose(out.to_numpy(), [[0.5, 0.5, 0.0]])

    def test_softmax_rows_and_temperature(self):
        scores = pd.DataFrame([[1.0, 2.0, 3.0]], columns=ACTIONS)
        cold, hot = ps.softmax(scores, 0.01), ps.softmax(scores, 100.0)
        assert cold.to_numpy().sum() == pytest.approx(1.0)
        assert cold.iloc[0, 2] == pytest.approx(1.0)
        np.testing.assert_allclose(hot.to_numpy(), 1 / 3, atol=0.01)

    def test_epsilon_greedy(self):
        out = ps.epsilon_greedy(pd.DataFrame([[0.0, 5.0, 1.0]], columns=ACTIONS), epsilon=0.3)
        np.testing.assert_allclose(out.to_numpy(), [[0.1, 0.8, 0.1]])

    def test_helpers_keep_labels_and_index(self):
        scores = pd.DataFrame([[0.0, 1.0, 2.0]], columns=ACTIONS, index=[42])
        out = ps.softmax(scores)
        assert list(out.columns) == ACTIONS and list(out.index) == [42]

    @pytest.mark.parametrize("bad", [dict(temperature=0), dict(epsilon=1.5)])
    def test_invalid_parameters(self, bad):
        scores = pd.DataFrame([[0.0, 1.0, 2.0]], columns=ACTIONS)
        fn = ps.softmax if "temperature" in bad else ps.epsilon_greedy
        with pytest.raises(ValueError):
            fn(scores, **bad)


class TestAlignment:
    def test_column_order_does_not_matter(self, setup):
        """Регрессия: в 0.2 переставленные колонки молча давали неверный ответ."""
        logs, data, probs = setup
        np.testing.assert_allclose(
            as_matrix(probs, logs, data), as_matrix(probs[ACTIONS[::-1]], logs, data)
        )

    def test_rows_aligned_by_index_labels(self, setup):
        logs, data, probs = setup
        shuffled = probs.sample(frac=1.0, random_state=1)
        np.testing.assert_allclose(as_matrix(shuffled, logs, data), probs.to_numpy())

    def test_range_index_is_positional(self, setup):
        logs, data, probs = setup
        logs = logs.set_axis(np.arange(100, 150))
        data = prepare_logs(logs, action="offer", reward="r", propensity="p", features=["x"])
        np.testing.assert_allclose(as_matrix(probs, logs, data), probs.to_numpy())

    def test_foreign_index_rejected(self, setup):
        logs, data, probs = setup
        with pytest.raises(ValueError, match="не сопоставляются"):
            as_matrix(probs.set_axis(np.arange(1000, 1050)), logs, data)

    def test_missing_logged_action_column_means_zero(self, setup):
        logs, data, _ = setup
        probs = pd.DataFrame({"card": 0.4, "deposit": 0.6}, index=logs.index)
        m = as_matrix(probs, logs, data)
        np.testing.assert_allclose(m[:, 2], 0.0)

    def test_new_action_with_mass_is_refused(self, setup):
        logs, data, probs = setup
        probs = probs * 0.9
        probs["mortgage"] = 0.1
        with pytest.raises(ValueError, match="нет в логах"):
            as_matrix(probs, logs, data)

    def test_new_action_without_mass_is_dropped(self, setup):
        logs, data, probs = setup
        probs = probs.assign(mortgage=0.0)
        assert as_matrix(probs, logs, data).shape == (50, 3)

    def test_label_type_mismatch_explained(self, setup):
        logs, data, probs = setup
        with pytest.raises(ValueError, match="типы меток"):
            as_matrix(probs.set_axis(["a", "b", "c"], axis=1), logs, data)

    def test_rows_must_sum_to_one(self, setup):
        logs, data, probs = setup
        with pytest.raises(ValueError, match="суммироваться в 1"):
            as_matrix(probs * 2, logs, data)


class TestInputKinds:
    def test_column_name_is_deterministic_policy(self, setup):
        logs, data, _ = setup
        m = as_matrix("chosen", logs, data)
        assert set(np.unique(m)) == {0.0, 1.0}
        assert [ACTIONS[i] for i in m.argmax(axis=1)] == list(logs["chosen"])

    def test_series_of_actions(self, setup):
        logs, data, _ = setup
        np.testing.assert_allclose(as_matrix(logs["chosen"], logs, data), as_matrix("chosen", logs, data))

    def test_object_with_action_probs_sees_no_reward(self, setup):
        logs, data, probs = setup
        seen = {}

        class Model:
            def action_probs(self, df):
                seen["columns"] = set(df.columns)
                return probs

        as_matrix(Model(), logs, data)
        assert "r" not in seen["columns"] and "p" not in seen["columns"]

    def test_callable(self, setup):
        logs, data, probs = setup
        np.testing.assert_allclose(as_matrix(lambda df: probs, logs, data), probs.to_numpy())

    def test_array_requires_integer_actions(self, setup):
        logs, data, probs = setup
        with pytest.raises(ValueError, match="без меток"):
            as_matrix(probs.to_numpy(), logs, data)

    def test_array_with_integer_actions(self):
        logs = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4], "a": [0, 1, 1, 0], "r": [1, 0, 1, 0]})
        data = prepare_logs(logs, action="a", reward="r", features=["x"])
        arr = np.array([[0.2, 0.8]] * 4)
        np.testing.assert_allclose(policy_matrix(arr, data, logs.drop(columns="r")), arr)

    def test_unknown_type(self, setup):
        logs, data, _ = setup
        with pytest.raises(TypeError):
            as_matrix(42, logs, data)


class TestShuffledLogs:
    """Регрессия: позиционная политика молча сопоставлялась по метке после sort/sample."""

    def make(self):
        rng = np.random.default_rng(3)
        logs = pd.DataFrame({
            "x": rng.normal(size=40), "offer": rng.choice(ACTIONS, 40), "p": 1 / 3,
            "r": rng.integers(0, 2, 40),
        }).sample(frac=1.0, random_state=0)  # индекс — перестановка 0..39
        data = prepare_logs(logs, action="offer", reward="r", propensity="p", features=["x"])
        probs = rng.dirichlet(np.ones(3), 40)  # строки в порядке логов
        return logs, data, probs

    def test_default_index_with_permuted_logs_is_ambiguous(self):
        logs, data, probs = self.make()
        with pytest.raises(ValueError, match="однозначно"):
            as_matrix(pd.DataFrame(probs, columns=ACTIONS), logs, data)

    def test_explicit_index_is_aligned(self):
        logs, data, probs = self.make()
        frame = pd.DataFrame(probs, columns=ACTIONS, index=logs.index)
        np.testing.assert_allclose(as_matrix(frame.sample(frac=1.0, random_state=1), logs, data), probs)

    def test_array_is_positional(self):
        logs = pd.DataFrame({"x": [0.1, 0.2, 0.3], "a": [0, 1, 0], "r": [1, 0, 1]}, index=[2, 0, 1])
        data = prepare_logs(logs, action="a", reward="r", features=["x"])
        arr = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
        np.testing.assert_allclose(policy_matrix(arr, data, logs.drop(columns="r")), arr)
