# FORGE marketing workspace

This directory has two layers:

- `release/` is the curated, reviewable public launch kit. Its manifest, release notes,
  and social copy are generated from verified claims already published in `README.md`.
- local benchmark checkpoints, profiler captures, screenshots, and raw logs are working
  inputs. Large weights plus `.log`/`.err` files remain ignored and must never be added to
  a public release.

Regenerate and validate the public layer with:

```bash
uv run python scripts/build_release_kit.py
uv run python scripts/build_release_kit.py --check
```

Do not manually add performance claims to the generated files. First land the validated
measurement in the README launch table, then regenerate this kit.
