from policyscope.validation import ValidationMode, run_simulation_validation


def test_validation_harness_runs_end_to_end_small():
    res = run_simulation_validation(
        n_runs=2,
        n_users=120,
        n_boot=10,
        estimators=("ips", "dr"),
        modes=(
            ValidationMode(name="estimated", propensity_source="estimated"),
            ValidationMode(name="logged", propensity_source="logged", propensity_col="propensity_A"),
        ),
    )
    assert len(res.run_rows) == 2 * 2 * 2
    assert not res.aggregate.empty


def test_validation_rows_include_oracle_and_metadata_fields():
    res = run_simulation_validation(
        n_runs=1,
        n_users=100,
        n_boot=10,
        estimators=("replay", "ips", "dm"),
        modes=(ValidationMode(name="estimated", propensity_source="estimated"),),
    )
    row = res.run_rows[0]
    assert row.oracle_v_b is not None
    assert row.oracle_delta is not None
    assert row.v_b_bias is not None
    assert row.delta_bias is not None
    d = res.to_dict()
    assert "aggregate" in d and isinstance(d["aggregate"], list)
