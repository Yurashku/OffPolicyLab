"""Тесты синтетической среды.

Если эталон неверен, все выводы валидационной обвязки бессмысленны, поэтому
оракул проверяется отдельно и независимо — сравнением с прямым сэмплированием.
"""

from __future__ import annotations

import numpy as np
import pytest

from policyscope.policies import make_policy
from policyscope.synthetic import SynthConfig, SyntheticRecommenderEnv


@pytest.fixture
def env() -> SyntheticRecommenderEnv:
    return SyntheticRecommenderEnv(SynthConfig(n_users=4000, seed=17))


class TestGeneration:
    def test_sample_users_has_expected_columns(self, env):
        X = env.sample_users()
        assert len(X) == 4000
        assert {"user_id", "loyal", "age", "risk", "income", "region"} <= set(X.columns)

    def test_logs_contain_actions_rewards_and_propensity(self, env):
        X = env.sample_users()
        logs = env.simulate_logs_A(make_policy("epsilon_greedy", seed=1, epsilon=0.2), X)
        assert {"a_A", "accept", "cltv", "propensity_A"} <= set(logs.columns)
        assert set(np.unique(logs["a_A"])) <= set(env.ACTIONS.tolist())
        assert ((logs["propensity_A"] > 0) & (logs["propensity_A"] <= 1)).all()

    def test_logged_propensity_matches_the_policy(self, env):
        """Записанное pi_A(a|x) должно совпадать с вероятностью выбранного действия."""
        X = env.sample_users()
        policy = make_policy("epsilon_greedy", seed=1, epsilon=0.2)
        logs = env.simulate_logs_A(policy, X)
        probs = policy.action_probs(X)
        expected = probs[np.arange(len(X)), logs["a_A"].to_numpy()]
        assert np.allclose(logs["propensity_A"].to_numpy(), expected)

    def test_action_frequencies_follow_the_policy(self, env):
        X = env.sample_users()
        policy = make_policy("epsilon_greedy", seed=1, epsilon=0.5)
        logs = env.simulate_logs_A(policy, X)
        observed = np.bincount(logs["a_A"], minlength=4) / len(X)
        expected = policy.action_probs(X).mean(axis=0)
        assert np.allclose(observed, expected, atol=0.03)


class TestOracle:
    def test_accept_oracle_matches_direct_sampling(self, env):
        """Аналитический оракул должен совпасть с прямым сэмплированием."""
        X = env.sample_users()
        policy = make_policy("softmax", seed=2, tau=0.8)
        exact = env.oracle_value(policy, X, metric="accept")

        rng = np.random.default_rng(99)
        cumulative = policy.action_probs(X).cumsum(axis=1)
        means = []
        for _ in range(40):
            a = (rng.random(len(X))[:, None] > cumulative).sum(axis=1).clip(0, 3)
            means.append(rng.binomial(1, env.true_accept_prob(X, a)).mean())
        mc_mean = float(np.mean(means))
        mc_se = float(np.std(means, ddof=1) / np.sqrt(len(means)))
        assert abs(exact - mc_mean) < 4 * mc_se

    def test_accept_oracle_is_deterministic(self, env):
        """Для accept оракул точен: повторный вызов даёт то же число."""
        X = env.sample_users()
        policy = make_policy("softmax", seed=2, tau=0.8)
        assert env.oracle_value(policy, X) == env.oracle_value(policy, X)

    def test_oracle_does_not_disturb_data_generation(self):
        """Вызов оракула не должен сдвигать генератор данных."""
        def logs_after(oracle_calls: int):
            env = SyntheticRecommenderEnv(SynthConfig(n_users=300, seed=5))
            X = env.sample_users()
            policy = make_policy("softmax", seed=2, tau=0.8)
            for _ in range(oracle_calls):
                env.oracle_value(policy, X, metric="accept")
            return env.simulate_logs_A(policy, X)["a_A"].to_numpy()

        assert np.array_equal(logs_after(0), logs_after(3))

    def test_deterministic_policy_oracle_equals_its_action_values(self, env):
        """Для детерминированной политики оракул = среднее p(x, argmax)."""
        X = env.sample_users()
        policy = make_policy("greedy", seed=2)
        chosen = policy.action_probs(X).argmax(axis=1)
        expected = float(np.mean(env.true_accept_prob(X, chosen)))
        assert env.oracle_value(policy, X, metric="accept") == pytest.approx(expected)

    def test_cltv_oracle_is_stable_across_calls(self, env):
        X = env.sample_users()
        policy = make_policy("softmax", seed=2, tau=0.8)
        a = env.oracle_value(policy, X, metric="cltv", n_mc=64)
        b = env.oracle_value(policy, X, metric="cltv", n_mc=64)
        assert a == pytest.approx(b, rel=0.005)

    def test_unknown_metric_rejected(self, env):
        X = env.sample_users(50)
        with pytest.raises(ValueError, match="метрика"):
            env.oracle_value(make_policy("greedy"), X, metric="no_such_metric")

    def test_better_policy_has_higher_oracle_value(self, env):
        """Политика, выбирающая argmax истинной вероятности, должна быть лучше случайной."""
        X = env.sample_users()

        class OracleBest:
            action_space = [0, 1, 2, 3]

            def action_probs(self, frame):
                best = env.accept_prob_all_actions(frame).argmax(axis=1)
                out = np.zeros((len(frame), 4))
                out[np.arange(len(frame)), best] = 1.0
                return out

        class Uniform:
            action_space = [0, 1, 2, 3]

            def action_probs(self, frame):
                return np.full((len(frame), 4), 0.25)

        assert env.oracle_value(OracleBest(), X) > env.oracle_value(Uniform(), X)


class TestActionDependentEffects:
    def test_effect_depends_on_action(self, env):
        """Иначе модель исхода без взаимодействий x*a была бы достаточной."""
        X = env.sample_users(500)
        probs = env.accept_prob_all_actions(X)
        # Разброс между действиями должен меняться от пользователя к пользователю.
        per_user_spread = probs.max(axis=1) - probs.min(axis=1)
        assert per_user_spread.std() > 0.01

    def test_best_action_varies_across_users(self, env):
        X = env.sample_users(500)
        best = env.accept_prob_all_actions(X).argmax(axis=1)
        assert len(np.unique(best)) > 1
