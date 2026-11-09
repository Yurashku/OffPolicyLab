# OffPolicyLab · Policyscope

[Подробное руководство на русском](README.ru.md)

[![CI](https://github.com/Yurashku/OffPolicyLab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Yurashku/OffPolicyLab/actions/workflows/ci.yml)

**Estimate how a new recommendation policy would perform using logs from an existing policy — and identify when those logs cannot support the comparison.**

Policyscope is a Python library for off-policy evaluation (OPE) in contextual bandit settings: selecting an offer, recommendation, or promotion. Given logged decisions from policy A and a candidate policy B, it estimates the change in expected reward, Δ = V(B) − V(A), with uncertainty and diagnostics.

The useful result is a decision report: an estimated effect, a confidence interval, agreement between estimators, and reasons to trust or withhold the conclusion. This helps decide which candidates warrant an online experiment. The causal interpretation still depends on the assumptions below.

**Package/import:** `policyscope` · **Python:** 3.10+ · **License:** MIT

## What you can inspect

| Engineering question | Implementation and evidence |
|---|---|
| How do estimates behave when the true effect is known? | An offer simulator with an analytic policy-value reference, [validation scenarios and recorded results](https://github.com/Yurashku/OffPolicyLab/blob/main/docs/validation.md). |
| How is uncertainty handled? | Cross-fitted nuisance models; cluster-aware inference; bootstrap refitting when propensities are estimated; [estimator rationale](https://github.com/Yurashku/OffPolicyLab/blob/main/docs/estimators.md). |
| What happens when the data are insufficient? | Support and weight diagnostics, estimator disagreement, placebo checks, and partial-identification bounds; [methodology](https://github.com/Yurashku/OffPolicyLab/blob/main/docs/methodology.md). |
| Can the same workflow handle changing production models? | Known logging policies, mixtures of model versions, candidate comparisons and selection; [monthly versions notebook](https://github.com/Yurashku/OffPolicyLab/blob/main/examples/monthly_versions.ipynb). |

The library computes DR, SNDR, IPS, SNIPS and DM estimates. Its default primary estimator is doubly robust (DR). Multiple metrics and candidate policies are supported, including multiplicity adjustment and simultaneous intervals. A narrow interval alone does not establish that an estimate is valid.

## Run a reproducible example

Install from the checked source revision in a virtual environment. This command uses Git; it does not assume a PyPI release.

```bash
python -m venv .venv
# Activate: source .venv/bin/activate (macOS/Linux)
# Activate: .\.venv\Scripts\Activate.ps1 (Windows PowerShell)
python -m pip install "policyscope @ git+https://github.com/Yurashku/OffPolicyLab.git@7ded1b1691c94babf531a23198720605cc29d781"
```

Save the following as `quickstart.py`, then run `python quickstart.py`. It creates synthetic customer logs with known action probabilities, evaluates a different offer policy, and compares the estimate with the simulator's true effect. No external dataset or account is needed.

```python
import policyscope as ps

sim = ps.synthetic.OfferSimulator(seed=0)
clients = sim.sample_clients(20_000)
policy_a = sim.policy("epsilon_greedy", epsilon=0.2, seed=1)
policy_b = sim.policy("softmax", temperature=1.0, seed=2)
logs = sim.simulate(clients, policy_a)

report = ps.evaluate(
    logs, policy_b,
    action="offer", reward=["accepted", "revenue"],
    propensity="propensity",
    features=["age", "income", "loyal", "risk", "region"],
    random_state=0, n_jobs=1,
)

for metric in ("accepted", "revenue"):
    result = report[metric]
    truth = (sim.true_value(clients, policy_b, metric)
             - sim.true_value(clients, policy_a, metric))
    print(f"{metric}: delta={result.delta.value:.4f}, "
          f"95% CI=[{result.delta.low:.4f}, {result.delta.high:.4f}], "
          f"truth={truth:.4f}, {result.decision}, {result.trust}")
```

Observed output from this revision on Windows with Python 3.14 (NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, SciPy 1.18.1):

```text
accepted: delta=0.0313, 95% CI=[0.0183, 0.0443], truth=0.0405, better, reliable
revenue: delta=14.1110, 95% CI=[11.0706, 17.1514], truth=15.5350, better, reliable
```

Here B improves the simulated acceptance rate by an estimated **3.13 percentage points** (95% CI: 1.83–4.43 pp); the simulator's true effect is 4.05 pp. Both metrics pass this run's diagnostics. Small numerical differences are possible with other dependency versions.

The `accepted` effect is a probability difference; multiply it by 100 for percentage points. `revenue` is measured in the simulator's monetary units per decision. The oracle uses the same sampled contexts and integrates over the outcome distribution. One run illustrates the API and an estimate-to-truth comparison; repeated simulation is needed to assess bias and interval coverage.

For a full report, use `print(report.to_markdown())`. Jupyter renders `report` directly. `report.summary()` returns a DataFrame, and `report.to_json("report.json")` writes a structured report. Human-readable report text is currently in Russian; the API and machine-readable status values use English identifiers.

## Use your own logs

One row represents one decision made by the logging policy. Prepare these fields:

| Field | Argument | Meaning |
|---|---|---|
| Logged action | `action=` | The action that A actually selected. |
| Observed reward(s) | `reward=` | One metric name or a list of names. |
| Logged propensity | `propensity=` | Probability that A selected the recorded action in that context, including exploration. Strongly preferred. |
| Pre-decision features | `features=` | Explicit list of context columns available before the action. |
| Repeated-observation ID | `cluster=` | Customer or other unit whose rows may be dependent. Required when a unit appears repeatedly. |

The candidate can be a probability DataFrame whose columns are action labels, a column/Series of candidate actions, or an object with `action_probs(df)`. For example, with columns `A_offer` and `B_offer` in a customer-month panel:

```python
ev = ps.Evaluator(
    df, action="A_offer", reward=["accepted", "revenue"],
    propensity="A_propensity", cluster="client_id",
    features=["age", "income", "loyal", "risk", "region"],
)
report = ev.evaluate("B_offer", name="B")
```

Use an evaluation period separate from the data used to train B. For panel data, the estimand is the average reward **per decision**: a customer observed six times contributes six rows. Clustering adjusts inference for dependence; it does not turn this into an equally weighted average per customer.

If propensities were not logged, the library can estimate them, but this adds modeling assumptions and uncertainty. Passing `logging_policy=` is another option when the actual historical policy is known. For versioned policies and their coverage restrictions, follow the [monthly versions example](https://github.com/Yurashku/OffPolicyLab/blob/main/examples/monthly_versions.ipynb).

## Read the decision and its limits

Each metric exposes `decision` (`better`, `worse`, `no_difference`, `inconclusive`, or `not_computed`) and `trust` (`reliable`, `caution`, or `unreliable`). Read the accompanying findings as well as the effect estimate.

- **Overlap matters.** If B chooses actions that A never took in the relevant contexts, the logs do not identify the full effect. Restrict the candidate to supported actions, collect exploration data, or examine bounds under explicit reward assumptions.
- **Features must be sufficient and measured before the decision.** Unobserved factors affecting both assignment and reward can invalidate causal interpretation. Diagnostics and placebo checks can reveal some failures; passing them does not prove the absence of confounding.
- **The setting is a contextual bandit.** Actions that change future states, retention, or which future observations exist require additional modeling. This is not a general sequential reinforcement-learning evaluator.
- **Uncertainty and identification are different.** A confidence interval does not cover arbitrary model misspecification or hidden bias. Failure to detect a difference does not establish equivalence.
- **Synthetic validation has a defined scope.** The repository includes difficult and deliberately failing scenarios. Recorded performance is evidence for those settings, not a guarantee on new business data.

## Explore further

- [Main tutorial](https://github.com/Yurashku/OffPolicyLab/blob/main/examples/tutorial.ipynb): end-to-end evaluation, diagnostics, and comparisons with synthetic truth.
- [Monthly model versions](https://github.com/Yurashku/OffPolicyLab/blob/main/examples/monthly_versions.ipynb): panels, version mixtures, coverage, bounds, and exploration.
- [Methodology](https://github.com/Yurashku/OffPolicyLab/blob/main/docs/methodology.md) and [estimator comparison](https://github.com/Yurashku/OffPolicyLab/blob/main/docs/estimators.md): assumptions and statistical choices.
- [Synthetic validation](https://github.com/Yurashku/OffPolicyLab/blob/main/docs/validation.md) and [public-data example](https://github.com/Yurashku/OffPolicyLab/blob/main/docs/real_data.md): evaluation procedures and limitations.
- [Detailed Russian usage guide](README.ru.md): the complete guide, including panel data, policy versions, diagnostics and further examples.
- [Changelog](https://github.com/Yurashku/OffPolicyLab/blob/main/CHANGELOG.md), [source](https://github.com/Yurashku/OffPolicyLab/tree/main/src/policyscope), and [tests](https://github.com/Yurashku/OffPolicyLab/tree/main/tests).

## Develop and validate

```bash
git clone https://github.com/Yurashku/OffPolicyLab.git
cd OffPolicyLab
python -m pip install -e ".[dev]"
python -m flake8 src tests examples
python -m pytest -m "not slow"
```

The [CI workflow](https://github.com/Yurashku/OffPolicyLab/blob/main/.github/workflows/ci.yml) also checks minimum dependency versions, builds and tests the wheel, executes both tutorials, and runs statistical validation. The simulation checks take longer:

```bash
python -m pytest -m slow
python examples/validate.py --runs 30 --clients 2000 --n-boot 60 --strict
```

Maintained by [Yurashku](https://github.com/Yurashku). The [commit history](https://github.com/Yurashku/OffPolicyLab/commits/main/) records contributions, including AI-assisted work credited in commit messages. Distributed under the [MIT license](https://github.com/Yurashku/OffPolicyLab/blob/main/LICENSE).
