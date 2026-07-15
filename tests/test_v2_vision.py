"""PRD-10: Multi-Encoder Vision Frontend tests."""

import torch


def _ensure_registry():
    """Import encoder modules to ensure they're registered."""
    import forge.vision.dinov2  # noqa: F401
    import forge.vision.siglip  # noqa: F401
    import forge.vision.theia  # noqa: F401


def test_vision_registry_register_and_create():
    """Can register and create vision encoders."""
    from forge.vision.registry import VisionEncoderRegistry

    _ensure_registry()
    encoder = VisionEncoderRegistry.create("siglip2-so400m", allow_mock=True)
    assert encoder is not None


def test_vision_registry_list():
    """Lists all registered encoders."""
    from forge.vision.registry import VisionEncoderRegistry

    _ensure_registry()
    encoders = VisionEncoderRegistry.list_encoders()
    assert "siglip2-so400m" in encoders
    assert "siglip-so400m" in encoders
    assert "dinov2-small" in encoders
    assert "theia-tiny" in encoders


def test_siglip_mock_output_shape():
    """SigLIP mock encoder produces correct output shape."""
    from forge.vision.siglip import MockSigLIPEncoder

    encoder = MockSigLIPEncoder(d_output=1152, n_tokens=729)
    images = torch.randn(2, 3, 384, 384)
    out = encoder(images)
    assert out.shape == (2, 729, 1152)


def test_dinov2_mock_output_shape():
    """DINOv2 mock encoder produces correct output shape."""
    from forge.vision.dinov2 import MockDINOv2Encoder

    encoder = MockDINOv2Encoder(d_output=384, n_tokens=729)
    images = torch.randn(2, 3, 518, 518)
    out = encoder(images)
    assert out.shape == (2, 729, 384)


def test_theia_mock_output_shape():
    """Theia mock encoder produces correct output shape."""
    from forge.vision.theia import MockTheiaEncoder

    encoder = MockTheiaEncoder(d_output=384, n_tokens=576)
    images = torch.randn(2, 3, 384, 384)
    out = encoder(images)
    assert out.shape == (2, 576, 384)


def test_multi_encoder_fusion_single():
    """Single encoder fusion produces correct output."""
    from forge.vision.multi_encoder import MultiEncoderFusion

    _ensure_registry()
    fusion = MultiEncoderFusion(
        encoder_names=["siglip2-so400m"],
        d_output=256,
        n_output_tokens=64,
        allow_mock=True,
    )
    images = torch.randn(2, 3, 384, 384)
    out = fusion(images)
    assert out.shape == (2, 64, 256)


def test_multi_encoder_fusion_multi():
    """Multi-encoder fusion produces correct output."""
    from forge.vision.multi_encoder import MultiEncoderFusion

    _ensure_registry()
    fusion = MultiEncoderFusion(
        encoder_names=["siglip-so400m", "theia-tiny"],
        d_output=256,
        n_output_tokens=64,
        allow_mock=True,
    )
    images = torch.randn(2, 3, 384, 384)
    out = fusion(images)
    assert out.shape == (2, 64, 256)


def test_vision_registry_refuses_implicit_mock() -> None:
    """Production registry calls fail instead of silently fabricating an encoder."""
    import pytest

    from forge.vision.registry import VisionEncoderRegistry

    _ensure_registry()
    with pytest.raises(FileNotFoundError, match="Canonical SigLIP2"):
        VisionEncoderRegistry.create("siglip2-so400m")
    with pytest.raises(RuntimeError, match="no configured production weights"):
        VisionEncoderRegistry.create("dinov2-small")


def test_bridge_auto_adapt_d_vision():
    """BridgeAttention adapts to different d_vision values."""
    from forge.modules.bridge_attention import BridgeAttention

    # Default d_vision=1152
    bridge1 = BridgeAttention(d_vision=1152, d_model=256, n_queries=16, n_heads=4, n_layers=1)
    vis1 = torch.randn(2, 100, 1152)
    out1 = bridge1(vis1)
    assert out1.shape == (2, 16, 256)

    # Different d_vision=384 (from DINOv2)
    bridge2 = BridgeAttention(d_vision=384, d_model=256, n_queries=16, n_heads=4, n_layers=1)
    vis2 = torch.randn(2, 100, 384)
    out2 = bridge2(vis2)
    assert out2.shape == (2, 16, 256)
