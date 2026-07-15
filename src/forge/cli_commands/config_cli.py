"""Starter configuration commands."""

from __future__ import annotations

import typer

config_app = typer.Typer(name="config", help="Create and inspect FORGE configuration")

STARTER_CONFIG = """# FORGE v3 starter configuration
# Run: forge pipeline --config forge.yaml --stage distill --skip-labels
paths:
  model_dir: ./models
  data_dir: ./data
  output_dir: ./outputs

student:
  variant: nano
  vision_encoder: google/siglip2-so400m-patch14-384
  language_model: Qwen/Qwen3-0.6B
  backbone_dtype: auto
  action_dim: 7
  action_head_type: diffusion

distill:
  max_steps: 2000
  batch_size: 16
  gradient_accumulation_steps: 4
  learning_rate: 0.0002

pruning:
  target_layers: 8

quant:
  method: qvla
  bits: 4

export:
  formats: [onnx, mlx, tensorrt]
  onnx_opset: 19
"""


@config_app.command("init")
def config_init() -> None:
    """Print a commented starter config to standard output."""
    typer.echo(STARTER_CONFIG, nl=False)


__all__ = ["STARTER_CONFIG", "config_app"]
