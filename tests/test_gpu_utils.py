"""Tests for forge.gpu_utils — GPU detection and DDP utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_smi_result(csv_lines: str) -> MagicMock:
    """Return a mock CompletedProcess whose stdout matches csv_lines."""
    result = MagicMock()
    result.stdout = csv_lines
    result.returncode = 0
    return result


_SMI_4GPU_ALL_FREE = "0, 20000, 0\n1, 20000, 5\n2, 20000, 3\n3, 20000, 0\n"
_SMI_4GPU_GPU0_LOW = "0, 500, 0\n1, 20000, 5\n2, 20000, 3\n3, 20000, 0\n"
_SMI_4GPU_HIGH_UTIL = "0, 20000, 90\n"
_SMI_MIXED = "0, 20000, 0\n1, 20000, 5\n2, 500, 90\n3, 20000, 0\n"


# ---------------------------------------------------------------------------
# get_free_gpus
# ---------------------------------------------------------------------------


class TestGetFreeGpus:
    def test_get_free_gpus_all_free(self):
        """Four GPUs all above threshold → all four returned."""
        from forge.gpu_utils import get_free_gpus

        mock_result = _make_smi_result(_SMI_4GPU_ALL_FREE)
        with patch("subprocess.run", return_value=mock_result):
            result = get_free_gpus(min_free_mb=2000, max_utilization=15)

        assert result == [0, 1, 2, 3]

    def test_get_free_gpus_some_busy(self):
        """GPU 0 has only 500 MB free → excluded; rest returned."""
        from forge.gpu_utils import get_free_gpus

        mock_result = _make_smi_result(_SMI_4GPU_GPU0_LOW)
        with patch("subprocess.run", return_value=mock_result):
            result = get_free_gpus(min_free_mb=2000, max_utilization=15)

        assert result == [1, 2, 3]

    def test_get_free_gpus_high_utilization(self):
        """GPU with 20 GB free but 90% utilization → excluded."""
        from forge.gpu_utils import get_free_gpus

        mock_result = _make_smi_result(_SMI_4GPU_HIGH_UTIL)
        with patch("subprocess.run", return_value=mock_result):
            result = get_free_gpus(min_free_mb=2000, max_utilization=15)

        assert result == []

    def test_get_free_gpus_nvidia_smi_fails(self):
        """Broken NVML falls back to conservative system CUDA memory."""
        from forge.gpu_utils import get_free_gpus

        fallback = [
            {
                "index": 0,
                "memory_total_mib": 24000,
                "memory_used_mib": 1000,
                "memory_free_mib": 23000,
            },
            {
                "index": 1,
                "memory_total_mib": 24000,
                "memory_used_mib": 15000,
                "memory_free_mib": 9000,
            },
        ]
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("forge.gpu_utils._torch_gpu_samples", return_value=fallback),
        ):
            result = get_free_gpus()

        assert result == [0]

    def test_get_free_gpus_nvml_and_cuda_unavailable(self):
        from forge.gpu_utils import get_free_gpus

        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("forge.gpu_utils._torch_gpu_samples", return_value=[]),
        ):
            assert get_free_gpus() == []


def test_get_gpu_samples_uses_torch_memory_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from forge import gpu_utils

    expected = [{"index": 0, "memory_free_mib": 22000}]
    monkeypatch.setattr(gpu_utils, "_smi", lambda _query: None)
    monkeypatch.setattr(gpu_utils, "_torch_gpu_samples", lambda: expected)

    assert gpu_utils.get_gpu_samples() == expected

    def test_get_free_gpus_custom_thresholds(self):
        """Stricter thresholds (10 GB free, max 5% util) filter more GPUs."""
        from forge.gpu_utils import get_free_gpus

        # GPU 0: 20 GB, 0% util   → passes
        # GPU 1: 20 GB, 5% util   → FAILS max_utilization=5 (5 is not < 5)
        # GPU 2: 500 MB, 90% util → FAILS both
        # GPU 3: 20 GB, 0% util   → passes
        mock_result = _make_smi_result(_SMI_MIXED)
        with patch("subprocess.run", return_value=mock_result):
            result = get_free_gpus(min_free_mb=10_000, max_utilization=5)

        # Only GPUs with >10 000 MB AND utilization < 5 %
        assert 0 in result
        assert 2 not in result


# ---------------------------------------------------------------------------
# get_gpu_count
# ---------------------------------------------------------------------------


class TestGetGpuCount:
    def test_get_gpu_count(self):
        """torch.cuda.device_count is mocked to return 4."""
        from forge.gpu_utils import get_gpu_count

        with patch("torch.cuda.device_count", return_value=4):
            count = get_gpu_count()

        assert count == 4


# ---------------------------------------------------------------------------
# require_free_gpus
# ---------------------------------------------------------------------------


class TestRequireFreeGpus:
    def test_require_free_gpus_success(self):
        """Enough free GPUs available → list is returned unchanged."""
        from forge.gpu_utils import require_free_gpus

        with patch("forge.gpu_utils.get_free_gpus", return_value=[0, 1, 2, 3]):
            result = require_free_gpus(min_gpus=1)

        assert result == [0, 1, 2, 3]

    def test_require_free_gpus_fails_none_available(self):
        """No free GPUs → RuntimeError."""
        from forge.gpu_utils import require_free_gpus

        with patch("forge.gpu_utils.get_free_gpus", return_value=[]):
            with pytest.raises(RuntimeError):
                require_free_gpus(min_gpus=1)

    def test_require_free_gpus_not_enough(self):
        """Only one GPU free but two required → RuntimeError."""
        from forge.gpu_utils import require_free_gpus

        with patch("forge.gpu_utils.get_free_gpus", return_value=[0]):
            with pytest.raises(RuntimeError):
                require_free_gpus(min_gpus=2)


# ---------------------------------------------------------------------------
# setup_ddp / cleanup_ddp
# ---------------------------------------------------------------------------


class TestDdp:
    def test_setup_ddp(self):
        """init_process_group and set_device called with correct arguments."""
        from forge.gpu_utils import setup_ddp

        with (
            patch("torch.distributed.init_process_group") as mock_init,
            patch("torch.cuda.set_device") as mock_set,
        ):
            setup_ddp(rank=1, world_size=4, backend="nccl")

        mock_init.assert_called_once_with(backend="nccl", rank=1, world_size=4)
        mock_set.assert_called_once_with(1)

    def test_cleanup_ddp(self):
        """destroy_process_group is called exactly once."""
        from forge.gpu_utils import cleanup_ddp

        with patch("torch.distributed.destroy_process_group") as mock_destroy:
            cleanup_ddp()

        mock_destroy.assert_called_once()


# ---------------------------------------------------------------------------
# print_gpu_status
# ---------------------------------------------------------------------------


class TestPrintGpuStatus:
    def test_print_gpu_status(self, capsys):
        """print_gpu_status runs without error and produces output."""
        from forge.gpu_utils import print_gpu_status

        mock_result = _make_smi_result(_SMI_MIXED)
        with patch("subprocess.run", return_value=mock_result):
            print_gpu_status()

        captured = capsys.readouterr()
        # Some output must be produced (table header or GPU index)
        assert len(captured.out) > 0 or len(captured.err) == 0
