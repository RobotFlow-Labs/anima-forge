"""Domain exceptions for truthful FORGE runtime failures."""

from __future__ import annotations

from pathlib import Path


class ForgeError(RuntimeError):
    """Base user-facing failure with an exact recovery command."""

    def __init__(self, message: str, *, hint: str) -> None:
        self.message = message
        self.hint = hint
        super().__init__(f"{message} {hint}")


class ForgeModelNotFoundError(ForgeError):
    """Raised when a required local model cannot be loaded safely."""

    def __init__(
        self,
        *,
        component: str,
        model_id: str,
        path: str | Path,
        cause: BaseException | None = None,
    ) -> None:
        self.component = component
        self.model_id = model_id
        self.path = Path(path)
        self.cause = cause

        detail = f" The loader reported: {cause}" if cause is not None else ""
        super().__init__(
            f"{component} weights are unavailable at {self.path}.{detail}",
            hint=(
                f"Fetch the required model with `forge models fetch {model_id}`, then "
                "verify the installation with `forge doctor`."
            ),
        )


class ForgeDataNotFoundError(ForgeError):
    """Raised when a required local dataset is absent."""

    def __init__(
        self,
        message: str,
        *,
        hint: str = "Generate real labels with `forge pipeline --stage labels`, then retry the command.",
    ) -> None:
        super().__init__(message, hint=hint)
