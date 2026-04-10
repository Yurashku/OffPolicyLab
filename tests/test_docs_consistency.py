from pathlib import Path


def test_architecture_doc_exists_and_mentions_domain_model():
    text = Path("docs/architecture.md").read_text(encoding="utf-8")
    assert "BanditSchema" in text
    assert "LoggedBanditDataset" in text
    assert "contextual bandit" in text.lower()
    assert "non-goals" in text.lower()


def test_agents_contains_stable_sections():
    text = Path("AGENTS.md").read_text(encoding="utf-8")
    assert "Scope проекта" in text
    assert "Терминология" in text
    assert "Архитектурные принципы" in text
    assert "Анти-паттерны для Codex" in text
