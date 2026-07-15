"""PRD-02: Student Architecture & Initialization.

FORGE student models: Nano (0.5B), Small (1.5B), Micro (0.2B).
Architecture: SigLIP2 (frozen) → Bridge Attention → current-gen LM (LoRA) → Action Head.

Usage:
    from forge.student import FORGEStudent
    from forge.config import ForgeConfig

    config = ForgeConfig.default()
    student = FORGEStudent(config.student)
    actions = student(images, language_ids)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from forge.config import StudentConfig
from forge.errors import ForgeModelNotFoundError
from forge.modules.action_head_factory import create_action_head
from forge.modules.bridge_attention import BridgeAttention
from forge.modules.lora import apply_lora

logger = logging.getLogger(__name__)


def _ensure_tokenizer_padding(tokenizer: Any, model: nn.Module) -> None:
    """Give decoder-only tokenizers a deterministic padding token or fail closed."""
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token is None or eos_token_id is None:
            raise RuntimeError("Language tokenizer defines neither a padding token nor a usable EOS token")
        tokenizer.pad_token = eos_token
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = eos_token_id
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise RuntimeError("Language tokenizer padding token could not be configured")
    model_config = getattr(model, "config", None)
    if model_config is not None:
        model_config.pad_token_id = pad_token_id


class FORGEStudent(nn.Module):
    """FORGE student VLA model.

    Architecture:
        SigLIP2-SO400M (frozen) → Bridge Attention → language model (LoRA) → Action Head

    The vision encoder is frozen to reduce trainable params by 80%.
    Only Bridge Attention, LoRA adapters, and Action Head are trained.
    """

    def __init__(self, config: StudentConfig, model_dir: str | Path | None = None):
        super().__init__()
        self.config = config
        self._model_dir: Path | None
        if model_dir is not None:
            self._model_dir = Path(model_dir).expanduser()
        elif config.allow_mock:
            # Preserve the explicit test/development path: no implicit heavy load.
            self._model_dir = None
        else:
            self._model_dir = Path(os.environ.get("FORGE_MODEL_DIR", "./models")).expanduser()
        self.vision_provenance = "unknown"
        self.language_provenance = "unknown"

        # PRD-25: AutoSense — detect model dimensions before building
        if self._model_dir and config.autosense:
            from forge.autosense import apply_autosense

            apply_autosense(config, self._model_dir)

        # Stage A: Vision Encoder (FROZEN)
        self.vision_encoder = self._load_vision_encoder()

        # Stage B: Bridge Attention (TRAINABLE)
        self.bridge = BridgeAttention(
            d_vision=config.bridge_d_vision,
            d_model=config.bridge_d_model,
            n_queries=config.bridge_n_queries,
            n_heads=config.bridge_n_heads,
            n_layers=config.bridge_n_layers,
        )

        # Stage C: Language Backbone with LoRA (PARTIALLY TRAINABLE)
        self.language, self.tokenizer = self._load_language_model()

        # Stage D: Action Head (TRAINABLE) — diffusion, flow, or chunk via config
        self.action_head = create_action_head(config)

        logger.info(f"FORGE-{config.variant} initialized:")
        logger.info(f"  Total params: {self.total_params / 1e6:.1f}M")
        logger.info(f"  Trainable params: {self.trainable_params / 1e6:.1f}M")
        logger.info(f"  Vision (frozen): {self._count_params(self.vision_encoder) / 1e6:.1f}M")
        logger.info(f"  Bridge: {self.bridge.param_count() / 1e6:.1f}M")
        logger.info(f"  Action Head: {self.action_head.param_count() / 1e6:.1f}M")

    def _load_vision_encoder(self) -> nn.Module:
        """Load and freeze SigLIP from local weights, or an explicitly allowed mock."""
        model_id = self.config.vision_encoder
        local_path = self._component_path(model_id)

        if local_path is None or not local_path.is_dir():
            if self.config.allow_mock:
                self.vision_provenance = "mock"
                logger.warning(
                    "Vision encoder weights unavailable at %s; explicit allow_mock is enabled",
                    local_path or "<model_dir not provided>",
                )
                return MockVisionEncoder(self.config.bridge_d_vision)
            raise self._model_error("Vision encoder", model_id, local_path)

        try:
            try:
                from transformers import SiglipVisionModel

                encoder = SiglipVisionModel.from_pretrained(str(local_path), local_files_only=True)
            except (AttributeError, OSError):
                from transformers import SiglipModel

                full_model = SiglipModel.from_pretrained(str(local_path), local_files_only=True)
                encoder = full_model.vision_model
                del full_model

            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()
            self.vision_provenance = "real"
            logger.info(f"Vision encoder loaded from {local_path}")
            return encoder

        except Exception as exc:
            if self.config.allow_mock:
                self.vision_provenance = "mock"
                logger.warning(
                    "Could not load SigLIP from %s; explicit allow_mock is enabled: %s",
                    local_path,
                    exc,
                )
                return MockVisionEncoder(self.config.bridge_d_vision)
            raise self._model_error(
                "Vision encoder",
                model_id,
                local_path,
                cause=exc,
            ) from exc

    def _load_language_model(self) -> tuple[nn.Module, object | None]:
        """Load the local causal LM with LoRA, or an explicitly allowed mock."""
        model_id = self.config.language_model
        local_path = self._component_path(model_id)

        if local_path is None or not local_path.is_dir():
            if self.config.allow_mock:
                self.language_provenance = "mock"
                logger.warning(
                    "Language model weights unavailable at %s; explicit allow_mock is enabled",
                    local_path or "<model_dir not provided>",
                )
                return MockLanguageModel(self.config.bridge_d_model), None
            raise self._model_error("Language model", model_id, local_path)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model: nn.Module = AutoModelForCausalLM.from_pretrained(
                str(local_path),
                dtype=self._backbone_dtype(),
                trust_remote_code=True,
                local_files_only=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(local_path),
                trust_remote_code=True,
                local_files_only=True,
            )
            _ensure_tokenizer_padding(tokenizer, model)

            # The language backbone is frozen by contract. ``apply_lora`` then
            # introduces the only trainable parameters inside this component.
            for parameter in model.parameters():
                parameter.requires_grad = False

            model = apply_lora(
                model,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
            )

            self.language_provenance = "real"
            logger.info(f"Language model loaded from {local_path} with LoRA (rank={self.config.lora_rank})")
            return model, tokenizer

        except Exception as exc:
            if self.config.allow_mock:
                self.language_provenance = "mock"
                logger.warning(
                    "Could not load language model from %s; explicit allow_mock is enabled: %s",
                    local_path,
                    exc,
                )
                return MockLanguageModel(self.config.bridge_d_model), None
            raise self._model_error(
                "Language model",
                model_id,
                local_path,
                cause=exc,
            ) from exc

    def _backbone_dtype(self) -> str | torch.dtype:
        """Resolve the configured frozen-backbone dtype for Transformers."""
        if self.config.backbone_dtype == "auto":
            return "auto"
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.config.backbone_dtype]

    @staticmethod
    def _module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
        """Return a module's floating parameter dtype without allocating tensors."""
        return next(
            (parameter.dtype for parameter in module.parameters() if parameter.is_floating_point()),
            fallback,
        )

    def _language_dtype(self, fallback: torch.dtype) -> torch.dtype:
        """Return the language embedding dtype used for ``inputs_embeds``."""
        language: Any = self.language
        if hasattr(language, "get_input_embeddings"):
            embeddings = language.get_input_embeddings()
            if embeddings is not None and hasattr(embeddings, "weight"):
                return embeddings.weight.dtype
        return self._module_dtype(self.language, fallback)

    def _component_path(self, model_id: str) -> Path | None:
        if self._model_dir is None:
            return None
        return self._model_dir / model_id.replace("/", "--")

    def _model_error(
        self,
        component: str,
        model_id: str,
        path: Path | None,
        *,
        cause: BaseException | None = None,
    ) -> ForgeModelNotFoundError:
        checked_path = path or Path("<model_dir not provided>") / model_id.replace("/", "--")
        return ForgeModelNotFoundError(
            component=component,
            model_id=model_id,
            path=checked_path,
            cause=cause,
        )

    @property
    def component_provenance(self) -> dict[str, str]:
        """Report whether each loaded student backbone is real or synthetic."""
        return {
            "vision": self.vision_provenance,
            "language": self.language_provenance,
        }

    def forward(
        self,
        images: torch.Tensor,
        language_ids: torch.Tensor | None = None,
        language_text: str | list[str] | None = None,
        proprioception: torch.Tensor | None = None,
        gt_actions: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass through the full student pipeline.

        Args:
            images: (B, C, H, W) input images — typically (B, 3, 384, 384)
            language_ids: (B, seq_len) tokenized language instruction
            language_text: Raw text instruction (tokenized internally if language_ids not provided)
            proprioception: (B, D_proprio) robot state (optional)
            gt_actions: (B, D_action) ground truth actions (training only)

        Returns:
            dict with 'actions' and optionally 'loss', 'vision_features'
        """
        batch_size = images.shape[0]
        if language_ids is not None and language_text is not None:
            raise ValueError("Provide language_ids or language_text, not both")
        if language_ids is None and language_text is not None:
            language_ids = self._tokenize_language_text(
                language_text,
                batch_size=batch_size,
                device=images.device,
            )

        # Vision: (B, C, H, W) → (B, N_vis, D_vision)
        vision_dtype = self._module_dtype(self.vision_encoder, images.dtype)
        vis_out = self.vision_encoder(images.to(dtype=vision_dtype))
        if hasattr(vis_out, "last_hidden_state"):
            vis_features = vis_out.last_hidden_state
        else:
            vis_features = vis_out

        # Bridge: (B, N_vis, D_vision) → (B, n_queries, D_model)
        bridge_dtype = self._module_dtype(self.bridge, vis_features.dtype)
        compressed_vis = self.bridge(vis_features.to(dtype=bridge_dtype))

        # Language embedding
        if language_ids is not None:
            lang_embeds = self._get_language_embeds(language_ids)
        else:
            lang_embeds = torch.zeros(
                batch_size,
                1,
                self.config.bridge_d_model,
                device=images.device,
                dtype=compressed_vis.dtype,
            )

        # Frozen v3 language backbones are bf16 while the trainable bridge and
        # action head remain fp32. Make both precision boundaries explicit.
        language_dtype = self._language_dtype(lang_embeds.dtype)
        compressed_vis = compressed_vis.to(dtype=language_dtype)
        lang_embeds = lang_embeds.to(dtype=language_dtype)

        # Concatenate: (B, n_queries + seq_len, D_model)
        combined = torch.cat([compressed_vis, lang_embeds], dim=1)

        # Language backbone processes combined sequence
        hidden = self._run_language_backbone(combined)

        # Pool vision tokens for action prediction
        action_features = hidden[:, : self.config.bridge_n_queries].mean(dim=1)  # (B, D_model)
        action_dtype = self._module_dtype(self.action_head, action_features.dtype)
        action_features = action_features.to(dtype=action_dtype)

        # Action head
        action_out = self.action_head(action_features, gt_actions=gt_actions)

        result = {
            "actions": action_out["actions"],
            "vision_features": compressed_vis,
        }
        if "loss" in action_out:
            result["loss"] = action_out["loss"]

        return result

    def _tokenize_language_text(
        self,
        language_text: str | list[str],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Tokenize raw instructions or fail closed when real tokenizer assets are absent."""
        instructions = [language_text] if isinstance(language_text, str) else list(language_text)
        if len(instructions) == 1 and batch_size > 1:
            instructions *= batch_size
        if len(instructions) != batch_size:
            raise ValueError(f"Received {len(instructions)} instructions for image batch {batch_size}")
        if any(not isinstance(instruction, str) or not instruction.strip() for instruction in instructions):
            raise ValueError("Every image requires a non-empty language instruction")

        if self.tokenizer is None:
            if self.language_provenance == "mock":
                return torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            raise RuntimeError(
                "The real language backbone has no tokenizer. Restore the mandatory local tokenizer "
                "with `forge models fetch --all-students`; FORGE will not ignore task instructions."
            )

        tokenizer: Any = self.tokenizer
        encoded = tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise RuntimeError("The mandatory tokenizer did not return a two-dimensional input_ids tensor")
        if input_ids.shape[0] != batch_size:
            raise RuntimeError(f"Tokenizer returned batch {input_ids.shape[0]} for image batch {batch_size}")
        return input_ids.to(device=device)

    def _get_language_embeds(self, language_ids: torch.Tensor) -> torch.Tensor:
        """Get language embeddings from backbone."""
        language: Any = self.language
        if hasattr(language, "get_input_embeddings"):
            embed_fn = language.get_input_embeddings()
            return embed_fn(language_ids)
        elif hasattr(language, "embed"):
            return language.embed(language_ids)
        else:
            # Mock fallback
            return torch.zeros(
                language_ids.shape[0],
                language_ids.shape[1],
                self.config.bridge_d_model,
                device=language_ids.device,
            )

    def _run_language_backbone(self, combined: torch.Tensor) -> torch.Tensor:
        """Run language backbone on combined vision + language tokens."""
        language: Any = self.language
        if hasattr(language, "model"):
            # HuggingFace CausalLM — use the inner model
            out = language.model(inputs_embeds=combined)
            return out.last_hidden_state
        elif hasattr(language, "forward_embeds"):
            return language.forward_embeds(combined)
        else:
            # Mock fallback
            return combined

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Get all trainable parameters (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

    @staticmethod
    def _count_params(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())


class MockVisionEncoder(nn.Module):
    """Mock vision encoder for testing without SigLIP weights."""

    def __init__(self, d_vision: int = 1152, n_tokens: int = 729):
        super().__init__()
        self.d_vision = d_vision
        self.n_tokens = n_tokens
        # Minimal conv to produce features
        self.proj = nn.Linear(3 * 16 * 16, d_vision)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        # Simple: reshape + project
        patches = images.unfold(2, 16, 16).unfold(3, 16, 16)  # (B, 3, H/16, W/16, 16, 16)
        n_patches = patches.shape[2] * patches.shape[3]
        patches = patches.contiguous().view(batch_size, 3, n_patches, 16, 16)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, n_patches, -1)

        # Pad or truncate to n_tokens
        if n_patches < self.n_tokens:
            padding = torch.zeros(batch_size, self.n_tokens - n_patches, patches.shape[-1], device=images.device)
            patches = torch.cat([patches, padding], dim=1)
        else:
            patches = patches[:, : self.n_tokens]

        return self.proj(patches)


class MockLanguageModel(nn.Module):
    """Mock language model for testing without Qwen weights."""

    def __init__(self, d_model: int = 896):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model)

    def forward_embeds(self, combined: torch.Tensor) -> torch.Tensor:
        return self.linear(combined)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = ids.shape
        return torch.randn(batch_size, sequence_length, self.d_model, device=ids.device)
