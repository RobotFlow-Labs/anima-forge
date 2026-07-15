"""Test for end-to-end pipeline runner."""

from pathlib import Path


def test_pipeline_end_to_end(tmp_path):
    """Run complete pipeline with mock data on CPU."""
    from forge.config import ForgeConfig
    from forge.pipeline import run_pipeline

    config = ForgeConfig.default()
    config.paths.data_dir = str(tmp_path / "data")
    config.paths.output_dir = str(tmp_path / "outputs")
    config.paths.model_dir = "/nonexistent"  # Force mocks
    config.student.bridge_d_vision = 128
    config.student.bridge_d_model = 64
    config.student.bridge_n_queries = 8
    config.student.bridge_n_heads = 4
    config.student.bridge_n_layers = 2
    config.student.action_head_layers = 2
    config.student.action_diffusion_steps = 3
    config.student.lora_rank = 4
    config.student.lora_alpha = 8
    config.distill.batch_size = 2
    config.distill.gradient_accumulation_steps = 1
    config.distill.warmup_steps = 2
    config.distill.save_every = 50
    config.pruning.target_layers = 4

    results = run_pipeline(
        config,
        device="cpu",
        skip_labels=True,
        max_distill_steps=10,
    )

    assert "total_time_seconds" in results
    assert results["total_time_seconds"] > 0

    # Check pipeline summary was saved
    summary = Path(config.paths.output_dir) / "pipeline_summary.json"
    assert summary.exists()


def test_pipeline_has_no_implicit_cuda_compression_fallback() -> None:
    source = Path("src/forge/pipeline.py").read_text(encoding="utf-8")
    assert "Retrying compression on CPU" not in source
    assert "Falling back to CPU quantization" not in source
    assert "_quantize_with_fallback" not in source
