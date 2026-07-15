"""Compatibility exports for trusted processors authored before Transformers 5."""

from __future__ import annotations

LEGACY_TOKENIZATION_EXPORTS = (
    "PaddingStrategy",
    "PreTokenizedInput",
    "TextInput",
    "TruncationStrategy",
)


def install_legacy_tokenization_exports() -> None:
    """Expose tokenization base symbols at their pre-5.x import location."""
    import transformers.tokenization_utils as legacy_module  # type: ignore[import-not-found]
    import transformers.tokenization_utils_base as base_module
    import transformers.tokenization_utils_sentencepiece as sentencepiece_module
    import transformers.tokenization_utils_tokenizers as tokenizers_module

    targets = (legacy_module, sentencepiece_module, tokenizers_module)
    for name in LEGACY_TOKENIZATION_EXPORTS:
        value = getattr(base_module, name)
        for target in targets:
            if not hasattr(target, name):
                setattr(target, name, value)


__all__ = ["install_legacy_tokenization_exports"]
