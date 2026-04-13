from pathlib import Path


def test_architecture_doc_exists_and_mentions_domain_model():
    text = Path("docs/architecture.md").read_text(encoding="utf-8")
    assert "BanditSchema" in text
    assert "LoggedBanditDataset" in text
    assert "contextual bandit" in text.lower()
    assert "non-goals" in text.lower()
    assert "significance" in text.lower()
    assert "scalar" in text.lower()
    assert "centered paired bootstrap" in text.lower()
    assert "ess" in text.lower()
    assert "replay overlap" in text.lower()
    assert "compare_policies" in text
    assert 'propensity_source="auto"' in text
    assert "logged vs estimated propensity" in text.lower()


def test_readme_mentions_p_value_method():
    text = Path("README.md").read_text(encoding="utf-8").lower()
    assert "centered paired bootstrap" in text
    assert "h0: delta = 0" in text
    assert "weight_ess_ratio" in text
    assert "compare_policies_multi_target" in text
    assert "propensity source modes" in text
    assert 'propensity_source="auto"' in text


def test_agents_contains_stable_sections():
    text = Path("AGENTS.md").read_text(encoding="utf-8")
    assert "Scope проекта" in text
    assert "Терминология" in text
    assert "Архитектурные принципы" in text
    assert "Анти-паттерны для Codex" in text
    assert "delta = V_B - V_A" in text
    assert "significance metadata" in text


def test_validation_harness_doc_exists():
    text = Path("docs/validation_harness.md").read_text(encoding="utf-8").lower()
    assert "run_simulation_validation" in text
    assert "oracle" in text
    assert "не гарантия" in text
