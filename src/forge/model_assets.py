"""Canonical Hugging Face model assets used by FORGE v3.

The runtime model registry tracks trained FORGE checkpoints. This module is a
separate manifest for upstream backbone and teacher weights so environment
health checks and asset downloads share one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelAsset:
    """An upstream model required or supported by FORGE."""

    repo_id: str
    role: str
    required: bool = True
    sidecar_url: str | None = None
    sidecar_filename: str | None = None
    sidecar_sha256: str | None = None
    config_filename: str = "config.json"
    weights_required: bool = True
    required_files: tuple[str, ...] = ()

    @property
    def local_name(self) -> str:
        """Return the canonical on-disk ``org--name`` directory name."""
        return self.repo_id.replace("/", "--")


# The eight assets required by the revised, GPU-verified v3 matrix. Qwen3.5
# remains an optional stretch target because its multimodal architecture is not
# yet a drop-in CausalLM backbone (PRD-44 R1b).
CORE_MODEL_ASSETS: tuple[ModelAsset, ...] = (
    ModelAsset("HuggingFaceTB/SmolLM2-135M", "student:micro"),
    ModelAsset("Qwen/Qwen3-0.6B", "student:nano"),
    ModelAsset("Qwen/Qwen3-1.7B", "student:small"),
    ModelAsset("Qwen/Qwen3-4B", "student:medium"),
    ModelAsset("google/siglip2-so400m-patch14-384", "vision"),
    ModelAsset("openvla/openvla-7b", "teacher"),
    ModelAsset(
        "robotics-diffusion-transformer/RDT2-FM",
        "teacher",
        sidecar_url=("https://ml.cs.tsinghua.edu.cn/~lingxuan/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt"),
        sidecar_filename="rdt2-umi-normalizer.pt",
        sidecar_sha256="03ebdd485e975630b02f2783b626dd146248f6f9bf51205a74406a0da4919728",
    ),
    ModelAsset("robotics-diffusion-transformer/RDT2-VQ", "teacher"),
    ModelAsset("lerobot/smolvla_base", "teacher"),
    ModelAsset("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "teacher"),
    ModelAsset("allenai/MolmoAct2-LIBERO-LeRobot", "teacher"),
    ModelAsset("allenai/MolmoAct2-LIBERO", "teacher"),
    ModelAsset(
        "allenai/MolmoAct2-FAST-Tokenizer",
        "teacher",
        config_filename="processor_config.json",
        weights_required=False,
        required_files=("tokenizer.json", "tokenizer_config.json"),
    ),
    ModelAsset("lerobot/VLA-JEPA-Pretrain", "teacher"),
    ModelAsset("Qwen/Qwen3-VL-2B-Instruct", "teacher"),
    ModelAsset("facebook/vjepa2-vitl-fpc64-256", "teacher"),
)

OPTIONAL_MODEL_ASSETS: tuple[ModelAsset, ...] = (
    ModelAsset("Qwen/Qwen3.5-0.8B", "student:stretch", required=False),
    ModelAsset("Qwen/Qwen3.5-2B", "student:stretch", required=False),
    ModelAsset("Qwen/Qwen3.5-4B", "student:stretch", required=False),
    ModelAsset("Qwen/Qwen2.5-0.5B", "student:legacy", required=False),
    ModelAsset("google/siglip-so400m-patch14-384", "vision:legacy", required=False),
)

ALL_MODEL_ASSETS: tuple[ModelAsset, ...] = CORE_MODEL_ASSETS + OPTIONAL_MODEL_ASSETS


def find_model_asset(name: str) -> ModelAsset | None:
    """Resolve a Hugging Face repo id or canonical local directory name."""
    normalized = name.strip().rstrip("/")
    for asset in ALL_MODEL_ASSETS:
        if normalized in {asset.repo_id, asset.local_name}:
            return asset
    return None


__all__ = [
    "ALL_MODEL_ASSETS",
    "CORE_MODEL_ASSETS",
    "OPTIONAL_MODEL_ASSETS",
    "ModelAsset",
    "find_model_asset",
]
