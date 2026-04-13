"""Run a lightweight synthetic validation experiment for OPE estimators."""

from __future__ import annotations

from policyscope.validation import ValidationMode, run_simulation_validation


if __name__ == "__main__":
    result = run_simulation_validation(
        n_runs=3,
        n_users=300,
        n_boot=20,
        estimators=("ips", "dr", "sndr", "switch_dr"),
        modes=(
            ValidationMode(name="estimated", propensity_source="estimated"),
            ValidationMode(name="logged", propensity_source="logged", propensity_col="propensity_A"),
            ValidationMode(name="crossfit", propensity_source="estimated", use_crossfit=True),
        ),
    )
    print(result.aggregate)
