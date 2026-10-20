"""Проверка согласованности документации с кодом.

Раньше эти тесты грепали markdown на подстроки — то есть проверяли, что документ
*содержит слова*, а не что слова *правдивы*. Документ, описывающий переименованный
параметр, проходил проверку. Здесь проверяется обратное направление: всё, что
упоминает документация, должно существовать в коде.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import policyscope
from policyscope.comparison import compare_policies
from policyscope.estimate import ALL_ESTIMATORS


def _read(repo_root: Path, name: str) -> str:
    path = repo_root / name
    if not path.exists():
        pytest.skip(f"{name} отсутствует")
    return path.read_text(encoding="utf-8")


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


class TestPublicApi:
    def test_all_exports_exist(self):
        missing = [n for n in policyscope.__all__ if not hasattr(policyscope, n)]
        assert not missing, f"объявлены в __all__, но отсутствуют: {missing}"

    def test_entry_point_importable_from_package_root(self):
        """Клиент не должен знать внутреннюю структуру модулей."""
        assert policyscope.compare_policies is compare_policies
        for name in ("OPEEvaluator", "TabularPolicy", "SoftmaxPolicy", "BanditSchema"):
            assert hasattr(policyscope, name)

    def test_version_matches_pyproject(self, repo_root):
        text = _read(repo_root, "pyproject.toml")
        declared = re.search(r'^version = "([^"]+)"', text, flags=re.MULTILINE)
        assert declared and declared.group(1) == policyscope.__version__


class TestReadmeExamples:
    def test_code_blocks_are_valid_python(self, repo_root):
        blocks = _python_blocks(_read(repo_root, "README.md"))
        assert blocks, "в README нет примеров кода"
        for i, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                pytest.fail(f"блок #{i} в README не парсится: {exc}")

    def test_referenced_policyscope_symbols_exist(self, repo_root):
        """Каждый ps.<symbol> из README должен существовать."""
        text = _read(repo_root, "README.md")
        referenced = set(re.findall(r"\bps\.([A-Za-z_][A-Za-z0-9_]*)", text))
        missing = [s for s in referenced if not hasattr(policyscope, s)]
        assert not missing, f"README ссылается на несуществующие символы: {missing}"

    def test_documented_compare_policies_arguments_exist(self, repo_root):
        """Имена аргументов, упомянутые в README как `arg=`, должны быть в сигнатуре."""
        text = _read(repo_root, "README.md")
        signature = set(inspect.signature(compare_policies).parameters)
        mentioned = set(re.findall(r"`(\w+)=`", text))
        missing = [m for m in mentioned if m not in signature]
        assert not missing, f"README упоминает несуществующие аргументы: {missing}"


class TestDocsReferences:
    @pytest.mark.parametrize(
        "doc", ["docs/architecture.md", "docs/validation_harness.md", "README.md"]
    )
    def test_no_unimplemented_estimators_advertised(self, repo_root, doc):
        text = _read(repo_root, doc)
        suspicious = {"ipw", "dr_os", "switch_ips", "cips", "dros"}
        found = {m for m in re.findall(r"`(\w+)`", text) if m.lower() in suspicious}
        assert not found, f"{doc} упоминает нереализованные оценщики: {found}"

    @pytest.mark.parametrize("doc", ["docs/architecture.md", "README.md"])
    def test_real_estimators_are_documented(self, repo_root, doc):
        text = _read(repo_root, doc)
        assert set(ALL_ESTIMATORS) & set(re.findall(r"`(\w+)`", text))

    def test_architecture_doc_names_real_modules(self, repo_root):
        text = _read(repo_root, "docs/architecture.md")
        modules = set(re.findall(r"`policyscope\.(\w+)`", text))
        import importlib

        for module in modules:
            importlib.import_module(f"policyscope.{module}")

    def test_no_references_to_removed_modules(self, repo_root):
        """После консолидации API ссылки на удалённые модули недопустимы."""
        removed = ["policyscope.bootstrap", "policyscope.ci", "estimator_with_bootstrap_ci"]
        for name in ("README.md", "docs/architecture.md", "docs/validation_harness.md"):
            text = _read(repo_root, name)
            found = [r for r in removed if r in text]
            assert not found, f"{name} ссылается на удалённое: {found}"


class TestDocstrings:
    @pytest.mark.parametrize("name", sorted(policyscope.__all__))
    def test_public_symbols_are_documented(self, name):
        obj = getattr(policyscope, name)
        if name == "__version__" or not (inspect.isclass(obj) or inspect.isfunction(obj)):
            pytest.skip("не вызываемый объект")
        assert obj.__doc__ and obj.__doc__.strip(), f"{name} без docstring"
