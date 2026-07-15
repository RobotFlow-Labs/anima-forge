#!/usr/bin/env python3
"""FORGE — Compress All Models for Edge Deployment.

3 compression strategies based on model type:
  1. Vision models (SAM2, DINOv2, DepthAnything, CLIP) → INT8 quantize + ONNX
  2. LLM/VLM models (Qwen, SmolLM, OpenELM) → INT4 quantize (NF4)
  3. Small models (YOLO, embeddings) → ONNX export only

Generates proper HF model cards with license, attribution, benchmarks.
Push to HuggingFace is MANUAL — review model cards first.

Usage:
    # Compress everything (no auto-push)
    uv run python scripts/compress_and_push.py

    # Specific category
    uv run python scripts/compress_and_push.py --category vision

    # Specific models
    uv run python scripts/compress_and_push.py --models "facebook--sam2.1-hiera-large"

    # Dry run
    uv run python scripts/compress_and_push.py --dry-run

    # Push after manual review
    uv run python scripts/compress_and_push.py --push
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("forge.compress")

HF_ORG = "robotflowlabs"
MODEL_DIR = Path(os.environ.get("FORGE_MODEL_DIR", "models"))

# ── Model Registry ───────────────────────────────────────────

ModelInfo = dict[str, Any]

MODELS: dict[str, ModelInfo] = {
    # === VISION MODELS → INT8 + ONNX ===
    "facebook--sam2.1-hiera-large": {
        "category": "vision",
        "hf_name": "sam2.1-hiera-large-int8",
        "arch": "sam2",
        "size_gb": 1.7,
        "input_shape": [1, 3, 1024, 1024],
        "license": "Apache-2.0",
        "source_repo": "facebook/sam2.1-hiera-large",
        "paper": "https://arxiv.org/abs/2408.00714",
        "description": "Segment Anything Model 2.1 (Hiera-Large) — video & image segmentation",
    },
    "facebook--sam2.1-hiera-small": {
        "category": "vision",
        "hf_name": "sam2.1-hiera-small-int8",
        "arch": "sam2",
        "size_gb": 0.35,
        "input_shape": [1, 3, 1024, 1024],
        "license": "Apache-2.0",
        "source_repo": "facebook/sam2.1-hiera-small",
        "paper": "https://arxiv.org/abs/2408.00714",
        "description": "Segment Anything Model 2.1 (Hiera-Small)",
    },
    "facebook--sam2.1-hiera-tiny": {
        "category": "vision",
        "hf_name": "sam2.1-hiera-tiny-int8",
        "arch": "sam2",
        "size_gb": 0.3,
        "input_shape": [1, 3, 1024, 1024],
        "license": "Apache-2.0",
        "source_repo": "facebook/sam2.1-hiera-tiny",
        "paper": "https://arxiv.org/abs/2408.00714",
        "description": "Segment Anything Model 2.1 (Hiera-Tiny)",
    },
    "facebook--dinov2-large": {
        "category": "vision",
        "hf_name": "dinov2-large-int8",
        "arch": "dinov2",
        "size_gb": 2.3,
        "input_shape": [1, 3, 518, 518],
        "license": "Apache-2.0",
        "source_repo": "facebook/dinov2-large",
        "paper": "https://arxiv.org/abs/2304.07193",
        "description": "DINOv2 ViT-L/14 — self-supervised vision features",
    },
    "depth-anything--Depth-Anything-V2-Large": {
        "category": "vision",
        "hf_name": "depth-anything-v2-large-int8",
        "arch": "depth_anything",
        "size_gb": 1.3,
        "input_shape": [1, 3, 518, 518],
        "license": "Apache-2.0",
        "source_repo": "depth-anything/Depth-Anything-V2-Large",
        "paper": "https://arxiv.org/abs/2406.09414",
        "description": "Depth Anything V2 Large — monocular depth estimation",
    },
    "depth-anything--Depth-Anything-V2-Small": {
        "category": "vision",
        "hf_name": "depth-anything-v2-small-int8",
        "arch": "depth_anything",
        "size_gb": 0.095,
        "input_shape": [1, 3, 518, 518],
        "license": "Apache-2.0",
        "source_repo": "depth-anything/Depth-Anything-V2-Small",
        "paper": "https://arxiv.org/abs/2406.09414",
        "description": "Depth Anything V2 Small — lightweight monocular depth",
    },
    "openai--clip-vit-large-patch14": {
        "category": "vision",
        "hf_name": "clip-vit-large-patch14-int8",
        "arch": "clip",
        "size_gb": 6.4,
        "input_shape": [1, 3, 224, 224],
        "license": "MIT",
        "source_repo": "openai/clip-vit-large-patch14",
        "paper": "https://arxiv.org/abs/2103.00020",
        "description": "CLIP ViT-L/14 — contrastive vision-language model",
    },
    "google--siglip-so400m-patch14-384": {
        "category": "vision",
        "hf_name": "siglip-so400m-patch14-384-int8",
        "arch": "siglip",
        "size_gb": 3.3,
        "input_shape": [1, 3, 384, 384],
        "license": "Apache-2.0",
        "source_repo": "google/siglip-so400m-patch14-384",
        "paper": "https://arxiv.org/abs/2303.15343",
        "description": "SigLIP SO400M — sigmoid loss vision-language encoder",
    },
    # === LLM/VLM → INT4 quantize ===
    "Qwen--Qwen2.5-7B-Instruct": {
        "category": "llm",
        "hf_name": "qwen2.5-7b-instruct-int4",
        "arch": "qwen2",
        "size_gb": 15,
        "license": "Apache-2.0",
        "source_repo": "Qwen/Qwen2.5-7B-Instruct",
        "paper": "https://arxiv.org/abs/2412.15115",
        "description": "Qwen2.5 7B Instruct — instruction-tuned LLM",
    },
    "Qwen--Qwen2.5-VL-7B-Instruct": {
        "category": "vlm",
        "hf_name": "qwen2.5-vl-7b-instruct-int4",
        "arch": "qwen2_vl",
        "size_gb": 16,
        "license": "Apache-2.0",
        "source_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
        "paper": "https://arxiv.org/abs/2502.13923",
        "description": "Qwen2.5-VL 7B — vision-language model with native resolution",
    },
    "Qwen--Qwen2.5-VL-3B-Instruct": {
        "category": "vlm",
        "hf_name": "qwen2.5-vl-3b-instruct-int4",
        "arch": "qwen2_vl",
        "size_gb": 7.1,
        "license": "Apache-2.0",
        "source_repo": "Qwen/Qwen2.5-VL-3B-Instruct",
        "paper": "https://arxiv.org/abs/2502.13923",
        "description": "Qwen2.5-VL 3B — compact vision-language model",
    },
    "HuggingFaceTB--SmolLM2-1.7B-Instruct": {
        "category": "llm",
        "hf_name": "smollm2-1.7b-instruct-int4",
        "arch": "llama",
        "size_gb": 22,
        "license": "Apache-2.0",
        "source_repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "paper": "https://huggingface.co/blog/smollm2",
        "description": "SmolLM2 1.7B Instruct — compact instruction-tuned LLM",
    },
    "apple--OpenELM-3B": {
        "category": "llm",
        "hf_name": "openelm-3b-int4",
        "arch": "openelm",
        "size_gb": 12,
        "license": "Apple Sample Code License",
        "source_repo": "apple/OpenELM-3B",
        "paper": "https://arxiv.org/abs/2404.14619",
        "description": "OpenELM 3B — efficient language model with layer-wise scaling",
    },
    # === SMALL / EXPORT ONLY ===
    "depth-anything--Video-Depth-Anything-Small": {
        "category": "small",
        "hf_name": "video-depth-anything-small-onnx",
        "arch": "depth_anything",
        "size_gb": 0.112,
        "input_shape": [1, 3, 518, 518],
        "license": "Apache-2.0",
        "source_repo": "depth-anything/Video-Depth-Anything-Small",
        "paper": "https://arxiv.org/abs/2406.09414",
        "description": "Video Depth Anything Small — temporal depth estimation",
    },
    "BAAI--bge-small-en-v1.5": {
        "category": "small",
        "hf_name": "bge-small-en-v1.5-onnx",
        "arch": "bert",
        "size_gb": 0.383,
        "license": "MIT",
        "source_repo": "BAAI/bge-small-en-v1.5",
        "paper": "https://arxiv.org/abs/2309.07597",
        "description": "BGE Small EN v1.5 — text embedding model",
    },
    "lerobot--smolvla_base": {
        "category": "small",
        "hf_name": "smolvla-base-onnx",
        "arch": "smolvla",
        "size_gb": 0.873,
        "license": "Apache-2.0",
        "source_repo": "lerobot/smolvla_base",
        "paper": "https://huggingface.co/lerobot/smolvla_base",
        "description": "SmolVLA Base — compact vision-language-action model",
    },
    "robotics-diffusion-transformer--RDT2-FM": {
        "category": "small",
        "hf_name": "rdt2-fm-onnx",
        "arch": "rdt2",
        "size_gb": 0.931,
        "license": "MIT",
        "source_repo": "robotics-diffusion-transformer/RDT2-FM",
        "paper": "https://arxiv.org/abs/2410.07864",
        "description": "RDT-2 Foundation Model — diffusion-based cross-embodiment policy",
    },
}


@dataclass
class CompressResult:
    model: str
    category: str
    hf_name: str
    status: str
    original_size_mb: float | None = None
    compressed_size_mb: float | None = None
    compression_ratio: float | None = None
    hf_url: str | None = None
    duration_s: float = 0.0
    error: str | None = None


def write_model_card(
    out_path: Path,
    model_name: str,
    model_info: dict,
    quant_type: str,
    metrics: dict,
) -> None:
    """Generate a proper HuggingFace model card with license and attribution."""
    hf_name = model_info["hf_name"]
    source = model_info.get("source_repo", model_name.replace("--", "/"))
    lic = model_info.get("license", "Unknown")
    paper = model_info.get("paper", "")
    desc = model_info.get("description", "")
    cat = model_info.get("category", "")
    orig_mb = metrics.get("original_size_mb", 0)
    comp_mb = metrics.get("compressed_size_mb", 0)
    ratio = round(orig_mb / max(comp_mb, 1), 1) if orig_mb > 0 else 0

    quant_label = {
        "int8": "INT8 Dynamic Quantization",
        "int4-nf4": "INT4 NF4 Double Quantization (bitsandbytes)",
        "export": "Original Weights (Edge Export)",
    }.get(quant_type, quant_type)

    tags = [
        "robotics",
        "edge-deployment",
        "anima",
        "forge",
        f"quantization-{quant_type}" if quant_type != "export" else "onnx-ready",
    ]
    if cat == "vision":
        tags.append("computer-vision")
    elif cat in ("llm", "vlm"):
        tags.extend(["nlp", "text-generation"])

    # YAML front matter
    lines = ["---"]
    lines.append(f"license: {lic}")
    lines.append(f"base_model: {source}")
    lines.append("tags:")
    for t in tags:
        lines.append(f"  - {t}")
    lines.append("library_name: transformers")
    lines.append("---")
    lines.append("")

    # Title
    lines.append(f"# {hf_name}")
    lines.append("")
    lines.append(f"> {desc}")
    lines.append("")
    lines.append(f"**{quant_label}** version of [`{source}`](https://huggingface.co/{source})")
    lines.append("for edge deployment in the [ANIMA](https://github.com/RobotFlow-Labs) robotics stack.")
    lines.append("")

    # Compression stats
    lines.append("## Compression")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Original size | {orig_mb:.0f} MB |")
    lines.append(f"| Compressed size | {comp_mb:.0f} MB |")
    lines.append(f"| Compression ratio | {ratio}x |")
    lines.append(f"| Quantization | {quant_label} |")
    lines.append("| Framework | PyTorch / ONNX |")
    lines.append("")

    # Usage
    lines.append("## Usage")
    lines.append("")
    if quant_type == "int4-nf4":
        lines.append("```python")
        lines.append("from transformers import AutoModelForCausalLM, AutoTokenizer")
        lines.append("")
        lines.append(f'model = AutoModelForCausalLM.from_pretrained("{HF_ORG}/{hf_name}", device_map="auto")')
        lines.append(f'tokenizer = AutoTokenizer.from_pretrained("{HF_ORG}/{hf_name}")')
        lines.append("```")
    elif quant_type == "int8":
        lines.append("```python")
        lines.append("import torch")
        lines.append('state_dict = torch.load("model_int8.pt", weights_only=True)')
        lines.append("# Load into original architecture, then use quantized weights")
        lines.append("```")
    else:
        lines.append("```python")
        lines.append("from transformers import AutoModel")
        lines.append(f'model = AutoModel.from_pretrained("{HF_ORG}/{hf_name}")')
        lines.append("```")
    lines.append("")

    # Attribution
    lines.append("## Attribution")
    lines.append("")
    lines.append(f"- **Original model**: [`{source}`](https://huggingface.co/{source})")
    lines.append(f"- **License**: {lic}")
    if paper:
        lines.append(f"- **Paper**: [{paper}]({paper})")
    lines.append(f"- **Compressed by**: [RobotFlowLabs](https://huggingface.co/{HF_ORG}) using FORGE")
    lines.append("")
    lines.append("## About FORGE")
    lines.append("")
    lines.append("FORGE is part of the ANIMA agentic robotics stack.")
    lines.append("It compresses large vision, language, and action models")
    lines.append("for real-time deployment on edge devices (Jetson, Raspberry Pi, Apple Silicon).")
    lines.append("")

    with open(out_path / "README.md", "w") as f:
        f.write("\n".join(lines))


def get_dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024**2), 1)


def _export_required_onnx(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    output_path: Path,
    model_name: str,
) -> None:
    """Export one required dynamic-batch ONNX artifact or fail the run."""
    try:
        batch = torch.export.Dim("batch", min=1)
        torch.onnx.export(
            model,
            (dummy,),
            str(output_path),
            input_names=["pixel_values"],
            output_names=["output"],
            dynamo=True,
            dynamic_shapes={"pixel_values": {0: batch}},
            opset_version=18,
        )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"ONNX exporter did not create a usable artifact at {output_path}")
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Required ONNX export failed for {model_name}: {exc}") from exc


def compress_vision_int8(model_name: str, model_info: dict, output_dir: Path) -> Path:
    """Quantize vision model to INT8 + export ONNX."""
    from transformers import AutoModel

    model_path = MODEL_DIR / model_name
    out_path = output_dir / model_info["hf_name"]
    out_path.mkdir(parents=True, exist_ok=True)

    arch = model_info["arch"]
    input_shape = model_info.get("input_shape", [1, 3, 224, 224])

    logger.info(f"  Loading {model_name} for INT8 quantization...")

    if arch == "clip":
        from transformers import CLIPModel

        model = CLIPModel.from_pretrained(str(model_path), local_files_only=True)
        vision_model = model.vision_model
        del model.text_model
    elif arch == "siglip":
        from transformers import SiglipModel

        full_model = SiglipModel.from_pretrained(str(model_path), local_files_only=True)
        vision_model = full_model.vision_model
        del full_model.text_model
    elif arch == "dinov2":
        from transformers import Dinov2Model

        vision_model = Dinov2Model.from_pretrained(str(model_path), local_files_only=True)
    elif arch == "depth_anything":
        # Check if HF format or raw .pth
        if (model_path / "config.json").exists():
            from transformers import AutoModelForDepthEstimation

            vision_model = AutoModelForDepthEstimation.from_pretrained(str(model_path), local_files_only=True)
        else:
            # Raw .pth checkpoint — load with torch directly
            pth_files = list(model_path.glob("*.pth"))
            if not pth_files:
                raise FileNotFoundError(f"No .pth files in {model_path}")
            logger.info(f"  Loading raw .pth checkpoint: {pth_files[0].name}")
            state_dict = torch.load(pth_files[0], map_location="cuda", weights_only=True)
            # Save as safetensors directly — can't ONNX export without model class
            from safetensors.torch import save_file

            out_path.mkdir(parents=True, exist_ok=True)
            save_file({k: v.contiguous() for k, v in state_dict.items()}, str(out_path / "model.safetensors"))
            for fname in ["config.json", "preprocessor_config.json", "README.md"]:
                src = model_path / fname
                if src.exists():
                    shutil.copy2(src, out_path / fname)
            write_model_card(
                out_path,
                model_name,
                model_info,
                "fp32",
                {
                    "original_size_mb": get_dir_size_mb(model_path),
                    "compressed_size_mb": get_dir_size_mb(out_path),
                },
            )
            orig = get_dir_size_mb(model_path)
            comp = get_dir_size_mb(out_path)
            logger.info(f"  Done: {orig:.1f}MB → {comp:.1f}MB (SafeTensors export)")
            return out_path
    elif arch == "sam2":
        from transformers import Sam2Model

        vision_model = Sam2Model.from_pretrained(str(model_path), local_files_only=True)
    else:
        vision_model = AutoModel.from_pretrained(str(model_path), local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_model.to(device)
    vision_model.eval()

    # INT8 quantization via torchao (GPU-native) or fallback to dynamic
    logger.info(f"  Quantizing to INT8 on {device}...")
    try:
        torchao = importlib.import_module("torchao")

        torchao.quantize_(vision_model, torchao.quantization.int8_dynamic_activation_int8_weight())
        torch.save(vision_model.state_dict(), out_path / "model_int8.pt")
    except Exception:
        # Fallback: dynamic quantization requires CPU
        vision_model.cpu()
        quantized = torch.quantization.quantize_dynamic(vision_model, {torch.nn.Linear}, dtype=torch.qint8)
        torch.save(quantized.state_dict(), out_path / "model_int8.pt")
        del quantized
        vision_model.to(device)

    # Export: ONNX for standard models, SafeTensors for SAM2 (roi_align not ONNX-compatible)
    skip_onnx = {"sam2"}
    if arch not in skip_onnx:
        onnx_path = out_path / "model.onnx"
        logger.info("  Exporting ONNX on GPU...")
        dummy = torch.randn(*input_shape, device=device)
        _export_required_onnx(vision_model, dummy, onnx_path, model_name)
        logger.info("  ONNX export done")
    else:
        # SAM2: TorchScript trace on GPU (custom ops not ONNX-compatible)
        try:
            logger.info("  Exporting TorchScript on GPU (custom ops)...")
            dummy = torch.randn(*input_shape, device=device)
            with torch.no_grad():
                traced = torch.jit.trace(vision_model, dummy)
            traced.save(str(out_path / "model.pt"))
            logger.info("  TorchScript export done — loads directly on GPU")
        except Exception as e:
            logger.warning(f"  TorchScript failed, saving SafeTensors instead: {e}")
            from safetensors.torch import save_file

            state_dict = {k: v.contiguous().cpu() for k, v in vision_model.state_dict().items()}
            save_file(state_dict, str(out_path / "model.safetensors"))

    vision_model.cpu()
    torch.cuda.empty_cache()

    # Copy config + preprocessor
    for fname in ["config.json", "preprocessor_config.json", "tokenizer_config.json"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, out_path / fname)

    # Write proper model card
    write_model_card(
        out_path,
        model_name,
        model_info,
        "int8",
        {
            "original_size_mb": get_dir_size_mb(MODEL_DIR / model_name),
            "compressed_size_mb": get_dir_size_mb(out_path),
        },
    )

    del vision_model
    gc.collect()
    torch.cuda.empty_cache()

    return out_path


def compress_llm_int4(model_name: str, model_info: dict, output_dir: Path) -> Path:
    """Quantize LLM/VLM to INT4 via bitsandbytes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_path = MODEL_DIR / model_name
    out_path = output_dir / model_info["hf_name"]
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Loading {model_name} in INT4...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    arch = model_info["arch"]

    model: Any
    if arch in ("qwen2_vl",):
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path),
            quantization_config=quant_config,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=quant_config,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )

    # Save quantized model
    logger.info("  Saving INT4 model...")
    model.save_pretrained(out_path)

    # Copy tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(out_path)
    except Exception as e:
        logger.warning(f"  Tokenizer save failed: {e}")
        # Copy tokenizer files manually
        for fname in model_path.glob("tokenizer*"):
            shutil.copy2(fname, out_path / fname.name)

    # Write proper model card
    write_model_card(
        out_path,
        model_name,
        model_info,
        "int4-nf4",
        {
            "original_size_mb": get_dir_size_mb(MODEL_DIR / model_name),
            "compressed_size_mb": get_dir_size_mb(out_path),
        },
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return out_path


def export_small_model(model_name: str, model_info: dict, output_dir: Path) -> Path:
    """Export small model — copy + add model card."""
    model_path = MODEL_DIR / model_name
    out_path = output_dir / model_info["hf_name"]
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Copying {model_name} for export...")

    # Copy all files
    for f in model_path.iterdir():
        if f.is_file():
            shutil.copy2(f, out_path / f.name)
        elif f.is_dir() and f.name not in (".git",):
            dst = out_path / f.name
            if not dst.exists():
                shutil.copytree(f, dst)

    # Write proper model card
    write_model_card(
        out_path,
        model_name,
        model_info,
        "export",
        {
            "original_size_mb": get_dir_size_mb(MODEL_DIR / model_name),
            "compressed_size_mb": get_dir_size_mb(out_path),
        },
    )

    return out_path


def push_to_hf(local_path: Path, hf_name: str) -> str:
    """Push compressed model to HuggingFace."""
    from huggingface_hub import HfApi

    repo_id = f"{HF_ORG}/{hf_name}"
    api = HfApi()

    logger.info(f"  Pushing to {repo_id}...")
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        commit_message=f"Compressed model: {hf_name}",
    )

    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"  Pushed: {url}")
    return url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FORGE — Compress All Models & Push to HF")
    parser.add_argument("--models", nargs="+", default=None, help="Specific model dir names")
    parser.add_argument(
        "--category",
        choices=["vision", "llm", "vlm", "small", "all"],
        default="all",
    )
    parser.add_argument("--output-dir", type=str, default="./outputs/compressed")
    parser.add_argument("--push", action="store_true", help="Push to HF (manual, review cards first)")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)

    # Select models
    if args.models:
        unknown = [name for name in args.models if name not in MODELS]
        if unknown:
            parser.error(f"unknown model name(s): {', '.join(unknown)}")
        selected = {k: v for k, v in MODELS.items() if k in args.models}
    elif args.category == "all":
        selected = MODELS
    else:
        selected = {k: v for k, v in MODELS.items() if v["category"] == args.category}

    # Verify models exist on disk
    available = {}
    missing = []
    for name, info in selected.items():
        if (MODEL_DIR / name).exists():
            available[name] = info
        else:
            missing.append(name)

    # Header
    print("=" * 70)
    print("FORGE — Compress & Push to HuggingFace")
    print("=" * 70)
    print(f"  Models:    {len(available)} available, {len(missing)} missing")
    print(f"  Output:    {output_dir}")
    print(f"  Push HF:   {'yes → ' + HF_ORG + '/' if args.push else 'no (use --push after review)'}")
    print()

    if missing:
        print(f"  Skipping {len(missing)} missing: {', '.join(missing)}")
        print()

    # Group by category
    by_cat: dict[str, list[tuple[str, ModelInfo]]] = {}
    for name, info in available.items():
        by_cat.setdefault(info["category"], []).append((name, info))

    for cat, models in sorted(by_cat.items()):
        print(f"  [{cat.upper()}] {len(models)} models:")
        for name, info in models:
            size = info.get("size_gb", "?")
            print(f"    {name} ({size}GB) → {HF_ORG}/{info['hf_name']}")
        print()

    if args.dry_run:
        print(f"[DRY RUN] Would compress {len(available)} models. Exiting.")
        return 1 if missing else 0

    # Process
    results: list[CompressResult] = []
    total_start = time.time()

    for i, (name, info) in enumerate(available.items(), 1):
        t0 = time.time()
        cat = info["category"]
        hf_name = info["hf_name"]

        logger.info(f"\n[{i}/{len(available)}] {name} ({cat})")

        result = CompressResult(
            model=name,
            category=cat,
            hf_name=hf_name,
            status="running",
        )

        try:
            original_size = get_dir_size_mb(MODEL_DIR / name)
            result.original_size_mb = original_size

            # Compress based on category
            if cat == "vision":
                out_path = compress_vision_int8(name, info, output_dir)
            elif cat in ("llm", "vlm"):
                out_path = compress_llm_int4(name, info, output_dir)
            elif cat == "small":
                out_path = export_small_model(name, info, output_dir)
            else:
                raise ValueError(f"Unknown category: {cat}")

            compressed_size = get_dir_size_mb(out_path)
            result.compressed_size_mb = compressed_size
            if original_size > 0:
                result.compression_ratio = round(original_size / max(compressed_size, 1), 1)

            # Push to HF (only if --push flag)
            if args.push:
                result.hf_url = push_to_hf(out_path, hf_name)

            result.status = "success"
            logger.info(f"  Done: {original_size}MB → {compressed_size}MB ({result.compression_ratio}x compression)")

        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)
            result.status = "failed"
            result.error = str(e)
            torch.cuda.empty_cache()
            gc.collect()

        result.duration_s = round(time.time() - t0, 1)
        results.append(result)

        # Save intermediate results
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "compress_results.json", "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

    total_time = time.time() - total_start

    # Final report
    print(f"\n{'=' * 80}")
    print("COMPRESSION RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Model':<45} {'Cat':<8} {'Orig(MB)':<10} {'Comp(MB)':<10} {'Ratio':<8} {'Status'}")
    print("-" * 80)

    for r in results:
        orig = f"{r.original_size_mb:.0f}" if r.original_size_mb else "--"
        comp = f"{r.compressed_size_mb:.0f}" if r.compressed_size_mb else "--"
        ratio = f"{r.compression_ratio:.1f}x" if r.compression_ratio else "--"
        icon = "✓" if r.status == "success" else "✗"
        print(f"  {icon} {r.model:<43} {r.category:<8} {orig:<10} {comp:<10} {ratio:<8} {r.status}")
        if r.hf_url:
            print(f"    → {r.hf_url}")

    print(f"{'=' * 80}")
    ok = len([r for r in results if r.status == "success"])
    fail = len([r for r in results if r.status == "failed"])
    print(f"\n{ok} succeeded, {fail} failed | Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    return 1 if fail or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
