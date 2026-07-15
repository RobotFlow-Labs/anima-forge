"""Real model tests — require GPU and downloaded models.

Skip with: pytest -m "not gpu"
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.gpu
def test_real_siglip_loading():
    """Load real SigLIP from disk."""
    pytest.importorskip("transformers")

    from pathlib import Path

    from forge.config import ForgeConfig

    config = ForgeConfig.default()
    siglip_path = Path(config.paths.model_dir) / config.paths.vision_encoder

    if not siglip_path.exists():
        pytest.skip(f"SigLIP weights not found at {siglip_path}")

    from transformers import SiglipVisionModel

    model = SiglipVisionModel.from_pretrained(str(siglip_path))
    assert model is not None

    # Test forward
    images = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        out = model(images)
    assert out.last_hidden_state.shape[0] == 1


@pytest.mark.gpu
def test_real_qwen_loading():
    """Load real Qwen2.5-0.5B from disk."""
    pytest.importorskip("transformers")

    from pathlib import Path

    from forge.config import ForgeConfig

    config = ForgeConfig.default()
    qwen_path = Path(config.paths.model_dir) / config.paths.language_model

    if not qwen_path.exists():
        pytest.skip(f"Qwen weights not found at {qwen_path}")

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(qwen_path))
    assert model is not None


@pytest.mark.gpu
def test_real_student_forward():
    """Real student forward pass on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig()
    model = FORGEStudent(config).cuda()
    model.eval()

    images = torch.randn(1, 3, 384, 384, device="cuda")
    lang_ids = torch.randint(0, 1000, (1, 10), device="cuda")

    with torch.no_grad():
        out = model(images, language_ids=lang_ids)

    assert "actions" in out
    assert out["actions"].device.type == "cuda"


@pytest.mark.gpu
def test_real_benchmark_latency():
    """Real latency benchmark on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from forge.benchmark.metrics import profile_latency
    from forge.config import StudentConfig
    from forge.student import FORGEStudent

    config = StudentConfig()
    model = FORGEStudent(config).cuda()

    metrics = profile_latency(model, n_warmup=3, n_samples=10, device="cuda")

    assert metrics.mean_ms > 0
    assert metrics.samples == 10
    # On GPU should be fast
    assert metrics.mean_ms < 5000  # Generous upper bound
