"""Benchmark 06: AutoSense Model Detection — scan all models on disk."""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from forge.benchmark.artifacts import write_json_artifact
from forge.benchmark.suites.runtime import results_dir

RESULTS_DIR = results_dir()

MODEL_DIR = Path(os.environ.get("FORGE_MODEL_DIR", "./models"))


def scan_model_configs(model_dir: Path) -> tuple[list[str], dict, dict, dict]:
    """Scan each config once and report only roles supported by its architecture metadata."""
    from forge.autosense import sense_language_model, sense_model_roles, sense_vision_encoder

    scanned_models: list[str] = []
    vision_results: dict = {}
    language_results: dict = {}
    scan_times: dict = {}

    for subdir in sorted(model_dir.iterdir()):
        if not subdir.is_dir():
            continue
        config_json = subdir / "config.json"
        if not config_json.exists():
            continue

        name = subdir.name
        scanned_models.append(name)
        roles = sense_model_roles(subdir)

        if "vision" in roles:
            t0 = time.perf_counter()
            vis = sense_vision_encoder(subdir)
            t_vis = (time.perf_counter() - t0) * 1000
            if vis and "d_output" in vis:
                vision_results[name] = vis
                scan_times[f"vision_{name}"] = round(t_vis, 2)
                n_tok = vis.get("n_tokens", "?")
                print(f"  Vision: {name} → d={vis['d_output']}, tokens={n_tok} ({t_vis:.1f}ms)")

        if "language" in roles:
            t0 = time.perf_counter()
            lang = sense_language_model(subdir)
            t_lang = (time.perf_counter() - t0) * 1000
            if lang and "d_model" in lang:
                language_results[name] = lang
                scan_times[f"language_{name}"] = round(t_lang, 2)
                print(f"  Language: {name} → d={lang['d_model']}, vocab={lang.get('vocab_size', '?')} ({t_lang:.1f}ms)")

    return scanned_models, vision_results, language_results, scan_times


def main():
    from forge.autosense import apply_autosense
    from forge.config import ForgeConfig

    print("=== AutoSense: Scanning all models on disk ===\n")

    if not MODEL_DIR.exists():
        print(f"SKIP: Model dir not found: {MODEL_DIR}")
        sys.exit(0)

    scanned_models, vision_results, language_results, scan_times = scan_model_configs(MODEL_DIR)

    # Test apply_autosense for different configurations
    print("\n=== Testing apply_autosense ===")
    config_tests = {}

    # Default v3 nano (Qwen3-0.6B)
    c1 = ForgeConfig.default()
    before_1 = {"bridge_d_vision": c1.student.bridge_d_vision, "bridge_d_model": c1.student.bridge_d_model}
    t0 = time.perf_counter()
    apply_autosense(c1.student, MODEL_DIR)
    t_apply = (time.perf_counter() - t0) * 1000
    after_1 = {"bridge_d_vision": c1.student.bridge_d_vision, "bridge_d_model": c1.student.bridge_d_model}
    config_tests["default_qwen3_06b"] = {
        "before": before_1,
        "after": after_1,
        "changed": before_1 != after_1,
        "time_ms": round(t_apply, 2),
    }
    print(f"  Default Qwen3-0.6B: {before_1} → {after_1} ({t_apply:.1f}ms)")

    # Canonical v3 small (Qwen3-1.7B)
    c2 = ForgeConfig.default()
    c2.student.language_model = "Qwen/Qwen3-1.7B"
    before_2 = {"bridge_d_vision": c2.student.bridge_d_vision, "bridge_d_model": c2.student.bridge_d_model}
    t0 = time.perf_counter()
    apply_autosense(c2.student, MODEL_DIR)
    t_apply = (time.perf_counter() - t0) * 1000
    after_2 = {"bridge_d_vision": c2.student.bridge_d_vision, "bridge_d_model": c2.student.bridge_d_model}
    config_tests["qwen3_17b"] = {
        "before": before_2,
        "after": after_2,
        "changed": before_2 != after_2,
        "time_ms": round(t_apply, 2),
    }
    print(f"  Qwen3-1.7B: {before_2} → {after_2} ({t_apply:.1f}ms)")

    results = {
        "benchmark": "autosense",
        "timestamp": datetime.now(UTC).isoformat(),
        "model_dir": MODEL_DIR.name,
        "n_models_scanned": len(scanned_models),
        "n_models_detected": len(set(vision_results) | set(language_results)),
        "unclassified_models": sorted(set(scanned_models) - set(vision_results) - set(language_results)),
        "vision_encoders": vision_results,
        "language_models": language_results,
        "scan_times_ms": scan_times,
        "config_tests": config_tests,
    }

    out_path = RESULTS_DIR / "bench_06_autosense.json"
    write_json_artifact(out_path, results)

    print(f"\nResults saved to {out_path}")
    print(f"  Vision encoders detected: {len(vision_results)}")
    print(f"  Language models detected: {len(language_results)}")
    print("BENCH 06: DONE")


if __name__ == "__main__":
    main()
