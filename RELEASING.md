# Как выпустить релиз

Релиз собирает и публикует [`.github/workflows/release.yml`](.github/workflows/release.yml) по тегу
вида `v0.3.0`. Порядок: сборка и smoke-тест → TestPyPI → PyPI → GitHub Release. Публикация идёт
через Trusted Publishing, без токенов.

## Один раз перед первым релизом

1. На [pypi.org](https://pypi.org/manage/account/publishing/) и
   [test.pypi.org](https://test.pypi.org/manage/account/publishing/) добавить «pending publisher»:
   - PyPI project name: `policyscope`;
   - owner: `Yurashku`, repository: `OffPolicyLab`;
   - workflow: `release.yml`;
   - environment: `pypi` на PyPI и `testpypi` на TestPyPI.
2. В настройках репозитория (Settings → Environments) создать окружения `testpypi` и `pypi`. Для
   `pypi` стоит включить «Required reviewers»: публикация будет ждать ручного подтверждения.

## Каждый релиз

1. CI на `main` зелёный, включая медленную валидацию.
2. Обновить версию в `src/policyscope/__init__.py` (`__version__`). Это единственный источник
   версии, `pyproject.toml` берёт её оттуда.
3. В `CHANGELOG.md` заменить «(не опубликована)» у новой версии на дату релиза.
4. Перед первым релизом в README заменить установку из GitHub на `pip install policyscope`.
5. Локально проверить сборку:

   ```bash
   python -m build && python -m twine check --strict dist/*
   ```

6. Закоммитить, затем поставить и отправить тег:

   ```bash
   git tag v0.3.0 && git push origin v0.3.0
   ```

7. Проследить за workflow `release`. После публикации проверить в чистом окружении:

   ```bash
   pip install policyscope==0.3.0
   ```

   Затем выполнить quickstart из README.
