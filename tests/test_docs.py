"""Документация не расходится с кодом: всё, что упоминают README и docs, существует."""

from __future__ import annotations

import ast
import inspect
import re

import pytest

import policyscope as ps


def _read(repo_root, name):
    return (repo_root / name).read_text(encoding="utf-8")


def _python_blocks(text):
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def test_public_api_exports_exist():
    assert all(hasattr(ps, name) for name in ps.__all__)


def test_version_has_single_source(repo_root):
    """Версия задаётся только в policyscope.__version__; pyproject и CHANGELOG с ней согласны."""
    text = _read(repo_root, "pyproject.toml")
    assert 'dynamic = ["version"]' in text
    assert 'version = { attr = "policyscope.__version__" }' in text
    assert re.fullmatch(r"\d+\.\d+\.\d+", ps.__version__)
    first = re.search(r"^## (\S+)", _read(repo_root, "CHANGELOG.md"), re.MULTILINE)
    assert first and first.group(1) == ps.__version__


@pytest.mark.parametrize("doc", ["README.md", "docs/methodology.md"])
def test_code_blocks_parse(repo_root, doc):
    for block in _python_blocks(_read(repo_root, doc)):
        ast.parse(block)


def test_readme_symbols_and_arguments_exist(repo_root):
    text = _read(repo_root, "README.md")
    missing = [s for s in set(re.findall(r"\bps\.([A-Za-z_]\w*)", text)) if not hasattr(ps, s)]
    assert not missing, f"README ссылается на несуществующие символы: {missing}"
    known = (set(inspect.signature(ps.Evaluator).parameters)
             | set(inspect.signature(ps.Evaluator.evaluate).parameters)
             | set(inspect.signature(ps.Evaluator.compare).parameters)
             | set(inspect.signature(ps.Evaluator.bounds).parameters)
             | set(inspect.signature(ps.evaluate).parameters))
    mentioned = set(re.findall(r"`(\w+)=`", text))
    assert mentioned <= known, f"README упоминает несуществующие аргументы: {mentioned - known}"


def test_quickstart_runs(repo_root):
    """Блок «Попробовать за минуту» исполняется как есть (на меньшей выборке)."""
    block = _python_blocks(_read(repo_root, "README.md"))[0].replace("20_000", "2_000")
    namespace: dict = {}
    exec(compile(block, "README", "exec"), namespace)
    assert namespace["report"].metrics


@pytest.mark.parametrize("doc", ["README.md", "docs/methodology.md", "docs/validation.md"])
def test_finding_codes_in_docs_exist(repo_root, doc):
    from policyscope import diagnostics, evaluator

    source = inspect.getsource(diagnostics) + inspect.getsource(evaluator)
    codes = set(re.findall(r"`([a-z]+(?:_[a-z]+)+)`", _read(repo_root, doc)))
    finding_like = {c for c in codes if c.endswith(("_logging", "_deficit", "_ess", "_rejected",
                                                    "_estimated", "_set", "_features",
                                                    "_probability"))}
    missing = [c for c in finding_like if f'"{c}"' not in source]
    assert not missing, f"{doc}: неизвестные коды находок {missing}"
