# Public release kit

The generated files in this directory are the launch inputs for the GitHub release and
outbound announcement copy. They intentionally contain only claims that already appear in
the public README benchmark table.

Before release:

1. Complete the remaining training, benchmark, export, and flagship evaluation gates.
2. Update the README only with validated artifact-backed numbers.
3. Run `uv run python scripts/build_release_kit.py`.
4. Run the full documentation/privacy checks and review the generated copy.
5. Use `release_notes.md` for the GitHub release and adapt `social_copy.md` per channel.

Raw `marketing/benchmarks/` checkpoints and `marketing/metrics/` logs are not release
assets and remain ignored.
