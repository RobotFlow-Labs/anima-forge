"""Production training CLI compatibility module."""

from forge.cli_commands.train_v3 import train_app, train_start, train_status, train_stop

__all__ = ["train_app", "train_start", "train_status", "train_stop"]
