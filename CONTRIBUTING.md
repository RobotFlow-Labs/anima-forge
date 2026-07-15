# Contributing to FORGE

Thanks for helping make VLA distillation reproducible and deployable.

1. Open an issue for substantial behavior changes.
2. Branch from `develop`; keep `main` release-only.
3. Install with `uv sync --locked --extra dev`.
4. Run `uv run ruff check src tests`, `uv run ruff format --check src tests`, and
   `uv run pytest -m "not gpu"` before opening a pull request.
5. Include real provenance for performance claims. Mock workflows must use explicit
   `--allow-mock` and must never be presented as validation evidence.

Bug reports should include `forge doctor --json`, the exact command, FORGE version,
platform, and a minimal redacted log. Never post tokens, private model URLs, datasets,
or robot credentials.

By contributing, you agree that your contribution is licensed under Apache-2.0 and
that you will follow the [Code of Conduct](CODE_OF_CONDUCT.md).
