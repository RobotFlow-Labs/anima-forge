# FORGE Model Directory

Weights live here (`./models/`, gitignored — only this README and `.gitkeep` are tracked).
Layout convention: `org--name/` (HF repo id with `/` → `--`).

Override location with `FORGE_MODEL_DIR` or `paths.model_dir` in your config YAML.

## Download everything

```bash
# from repo root — uses HF_TOKEN from .env if set
for repo in google/siglip2-so400m-patch14-384 Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B \
            HuggingFaceTB/SmolLM2-135M lerobot/smolvla_base \
            robotics-diffusion-transformer/RDT2-FM openvla/openvla-7b; do
  HF_HUB_DISABLE_XET=1 hf download "$repo" --local-dir "models/${repo//\//--}"
done
```

`forge models fetch --all` automates downloads for the supported registry.

## Student backbones (2026 refresh — Qwen3 dense, Apache-2.0, GPU-verified drop-in)

| Variant | Model | HF link | Size |
|---------|-------|---------|------|
| micro | SmolLM2-135M | https://huggingface.co/HuggingFaceTB/SmolLM2-135M | ~270 MB |
| nano *(default)* | Qwen3-0.6B | https://huggingface.co/Qwen/Qwen3-0.6B | ~1.5 GB |
| small | Qwen3-1.7B | https://huggingface.co/Qwen/Qwen3-1.7B | ~3.5 GB |
| medium | Qwen3-4B | https://huggingface.co/Qwen/Qwen3-4B | ~8 GB |

Legacy (v2-era, still supported via config): Qwen2.5-0.5B / 1.5B / 3B.

## Vision encoder

| Model | HF link | Notes |
|-------|---------|-------|
| SigLIP2-SO400M patch14-384 | https://huggingface.co/google/siglip2-so400m-patch14-384 | 2026 default — same dims as v1 SigLIP (d=1152, 729 tokens), drop-in |
| SigLIP-SO400M patch14-384 (legacy) | https://huggingface.co/google/siglip-so400m-patch14-384 | v2-era default |

## Teachers

| Teacher | HF link | Size | Type |
|---------|---------|------|------|
| OpenVLA-7B | https://huggingface.co/openvla/openvla-7b | ~15 GB | token-AR, H=1 |
| RDT2-FM | https://huggingface.co/robotics-diffusion-transformer/RDT2-FM | ~1 GB | diffusion, H=8 |
| SmolVLA-base | https://huggingface.co/lerobot/smolvla_base | ~1 GB | parallel, H=1 |

The full required teacher fleet and companion assets are validated by `forge doctor`.

## Rules

- Never commit weights. `.gitignore` covers `/models/` (this README + `.gitkeep` excepted).
- Check disk before big downloads: `df -h .` (need ~40 GB for the full set).
- Symlinks into an HF cache (`.hf-cache/hub/*/snapshots/<sha>`) are valid entries too.
