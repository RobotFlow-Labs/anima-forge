from __future__ import annotations

import json

from typer.testing import CliRunner

from forge.benchmark.comparison import build_comparison, compare_reports


def test_comparison_payload_has_deltas_and_preserves_unmeasured_metrics() -> None:
    payload = build_comparison(
        [
            {"model_name": "a", "latency": {"mean_ms": 10.0}},
            {"model_name": "b", "latency": {"mean_ms": 8.0}},
        ]
    )

    latency = payload["metrics"]["latency.mean_ms"]
    assert latency["values"] == [10.0, 8.0]
    assert latency["delta_report2_minus_report1"] == -2.0
    assert latency["relative_change_pct"] == -20.0
    assert payload["metrics"]["throughput.actions_per_second"]["values"] == [None, None]


def test_plain_comparison_never_renders_missing_values_as_zero(capsys, monkeypatch) -> None:
    import forge.benchmark.comparison as comparison

    def unavailable(*args, **kwargs):
        raise ImportError

    monkeypatch.setattr(comparison, "_compare_rich", unavailable)
    compare_reports([{"model_name": "a"}, {"model_name": "b"}])
    output = capsys.readouterr().out
    assert "Not measured" in output
    assert "latency=0" not in output.lower()


def test_benchmark_compare_json_emits_structured_delta(tmp_path) -> None:
    from forge.cli_v2 import benchmark_app

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"model_name": "a", "latency": {"mean_ms": 10}}', encoding="utf-8")
    second.write_text('{"model_name": "b", "latency": {"mean_ms": 8}}', encoding="utf-8")

    result = CliRunner().invoke(benchmark_app, ["compare", str(first), str(second), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["reports"][0]["model_name"] == "a"
    assert payload["metrics"]["latency.mean_ms"]["delta_report2_minus_report1"] == -2.0
