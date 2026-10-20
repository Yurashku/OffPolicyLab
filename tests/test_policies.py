"""Тесты политик и адаптеров: свойства распределений и устойчивость к подвыборкам."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from policyscope.policies import (
    DeterministicPolicy,
    EpsilonGreedyPolicy,
    SoftmaxPolicy,
    TabularPolicy,
    make_policy,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "s0": rng.normal(size=50),
        "s1": rng.normal(size=50),
        "s2": rng.normal(size=50),
    })


class TestProbabilityAxioms:
    """Строки любой политики обязаны быть корректным распределением."""

    def test_tabular_rows_sum_to_one(self, frame):
        probs = np.full((len(frame), 3), 1 / 3)
        out = TabularPolicy(probs, index=frame.index).action_probs(frame)
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_softmax_rows_sum_to_one(self, frame):
        out = SoftmaxPolicy(["s0", "s1", "s2"]).action_probs(frame)
        assert np.allclose(out.sum(axis=1), 1.0)
        assert (out > 0).all()

    def test_deterministic_rows_are_one_hot(self, frame):
        actions = np.array([0, 1, 2] * 16 + [0, 1])
        out = DeterministicPolicy(actions, action_space=[0, 1, 2], index=frame.index)
        probs = out.action_probs(frame)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert set(np.unique(probs)) <= {0.0, 1.0}

    def test_epsilon_greedy_gives_eps_over_k_to_non_greedy(self, frame):
        eps, k = 0.3, 3
        probs = EpsilonGreedyPolicy(["s0", "s1", "s2"], epsilon=eps).action_probs(frame)
        assert np.allclose(probs.sum(axis=1), 1.0)
        greedy = probs.max(axis=1)
        assert np.allclose(greedy, 1 - eps + eps / k)
        non_greedy = np.sort(probs, axis=1)[:, :-1]
        assert np.allclose(non_greedy, eps / k)

    def test_epsilon_one_gives_uniform(self, frame):
        probs = EpsilonGreedyPolicy(["s0", "s1", "s2"], epsilon=1.0).action_probs(frame)
        assert np.allclose(probs, 1 / 3)


class TestTemperature:
    def test_higher_temperature_moves_towards_uniform(self, frame):
        cold = SoftmaxPolicy(["s0", "s1", "s2"], temperature=0.1).action_probs(frame)
        hot = SoftmaxPolicy(["s0", "s1", "s2"], temperature=20.0).action_probs(frame)
        assert hot.max(axis=1).mean() < cold.max(axis=1).mean()
        assert hot.max(axis=1).mean() == pytest.approx(1 / 3, abs=0.05)

    def test_non_positive_temperature_rejected(self):
        with pytest.raises(ValueError):
            SoftmaxPolicy(["s0"], temperature=0.0)


class TestRowAlignment:
    """Cross-fitting и bootstrap передают политике подвыборки строк."""

    def test_indexed_policy_survives_subsetting(self, frame):
        probs = np.random.default_rng(1).dirichlet(np.ones(3), size=len(frame))
        policy = TabularPolicy(probs, index=frame.index)
        subset = frame.iloc[[5, 2, 40]]
        assert np.allclose(policy.action_probs(subset), probs[[5, 2, 40]])

    def test_indexed_policy_survives_duplicated_rows(self, frame):
        probs = np.random.default_rng(2).dirichlet(np.ones(3), size=len(frame))
        policy = TabularPolicy(probs, index=frame.index)
        resampled = frame.take([3, 3, 3, 7])
        assert np.allclose(policy.action_probs(resampled), probs[[3, 3, 3, 7]])

    def test_positional_policy_rejects_subsetting_with_clear_message(self, frame):
        probs = np.full((len(frame), 3), 1 / 3)
        policy = TabularPolicy(probs)
        with pytest.raises(ValueError, match="index=df.index"):
            policy.action_probs(frame.iloc[:10])

    def test_column_based_policy_needs_no_index(self, frame):
        policy = SoftmaxPolicy(["s0", "s1", "s2"])
        assert policy.action_probs(frame.iloc[:10]).shape == (10, 3)

    def test_duplicate_index_rejected(self, frame):
        probs = np.full((len(frame), 3), 1 / 3)
        bad_index = pd.Index([0] * len(frame))
        with pytest.raises(ValueError, match="уникальным"):
            TabularPolicy(probs, index=bad_index)


class TestValidation:
    def test_action_space_length_must_match(self, frame):
        with pytest.raises(ValueError, match="action_space"):
            TabularPolicy(np.full((len(frame), 3), 1 / 3), action_space=["a", "b"])

    def test_nan_probabilities_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            TabularPolicy(np.array([[np.nan, 1.0]]))

    def test_unknown_action_rejected(self, frame):
        policy = DeterministicPolicy(
            np.array(["z"] * len(frame)), action_space=["a", "b"], index=frame.index
        )
        with pytest.raises(ValueError, match="action_space"):
            policy.action_probs(frame)

    def test_epsilon_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="epsilon"):
            EpsilonGreedyPolicy(["s0"], epsilon=1.5)


class TestActionArgmax:
    def test_argmax_returns_labels_not_indices(self, frame):
        probs = np.tile([0.1, 0.7, 0.2], (len(frame), 1))
        policy = TabularPolicy(probs, action_space=["x", "y", "z"], index=frame.index)
        assert set(policy.action_argmax(frame)) == {"y"}


class TestSyntheticPolicies:
    @pytest.fixture
    def users(self):
        from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv

        return SyntheticRecommenderEnv(SynthConfig(n_users=200, seed=3)).sample_users()

    @pytest.mark.parametrize("kind", ["greedy", "epsilon_greedy", "softmax"])
    def test_synthetic_policies_are_valid_distributions(self, users, kind):
        probs = make_policy(kind, seed=1).action_probs(users)
        assert probs.shape == (len(users), 4)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all()

    def test_epsilon_greedy_has_full_support(self, users):
        """Без положительной вероятности у каждого действия веса вырождаются."""
        probs = make_policy("epsilon_greedy", seed=1, epsilon=0.2).action_probs(users)
        assert (probs > 0).all()

    def test_greedy_is_deterministic(self, users):
        probs = make_policy("greedy", seed=1).action_probs(users)
        assert set(np.unique(probs)) <= {0.0, 1.0}

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="Неизвестный тип"):
            make_policy("no_such_policy")  # type: ignore[arg-type]
