"""Foundation tests — verify project setup, imports, config, backend detection."""

from pathlib import Path


def test_forge_imports():
    """Verify core forge package imports."""
    import forge

    assert forge.__version__ == "3.0.1"


def test_backend_detection():
    """Verify backend auto-detection works."""
    from forge.backend import BackendType, detect_backend

    backend = detect_backend()
    assert backend in (BackendType.CUDA, BackendType.MLX, BackendType.CPU)


def test_backend_instance():
    """Verify backend singleton creates and returns device info."""
    from forge.backend import get_backend, reset_backend

    reset_backend()
    backend = get_backend()
    info = backend.get_device_info()
    assert info.device_name is not None
    assert info.vram_gb >= 0


def test_config_default():
    """Verify default config loads with sane values."""
    from forge.config import ForgeConfig

    cfg = ForgeConfig.default()
    assert cfg.student.variant == "nano"
    assert cfg.student.action_dim == 7
    assert cfg.distill.temperature == 4.0
    assert cfg.pruning.target_layers == 8
    assert cfg.quant.target_avg_bits == 4.0


def test_config_from_yaml():
    """Verify YAML config loading."""
    from forge.config import ForgeConfig

    cfg_path = Path("configs/forge_nano.yaml")
    if cfg_path.exists():
        cfg = ForgeConfig.from_yaml(cfg_path)
        assert cfg.student.variant == "nano"
        assert cfg.student.lora_rank == 32


def test_model_paths_exist():
    """Model IDs resolve to the documented local directory convention."""
    from forge.config import ForgeConfig

    cfg = ForgeConfig.default()
    model_dir = Path(cfg.paths.model_dir)

    assert cfg.paths.teacher_path == model_dir / cfg.paths.teacher
    assert cfg.paths.vision_encoder_path == model_dir / cfg.paths.vision_encoder
    assert cfg.paths.language_model_path == model_dir / cfg.paths.language_model


def test_backend_zeros():
    """Verify tensor creation on detected backend."""
    from forge.backend import get_backend, reset_backend

    reset_backend()
    backend = get_backend()
    t = backend.zeros(2, 3)
    arr = backend.to_numpy(t)
    assert arr.shape == (2, 3)
    assert arr.sum() == 0.0
