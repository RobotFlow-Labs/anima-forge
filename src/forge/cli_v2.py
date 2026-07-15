"""FORGE v2 enhanced CLI."""

from __future__ import annotations

import typer

from forge.cli_commands import (
    benchmark_app,
    config_app,
    curriculum_app,
    demo_app,
    embodiment_app,
    eval_app,
    finetune_app,
    hyperparam_app,
    metrics_app,
    models_app,
    profile_app,
    quantize_app,
    students_app,
    teacher_app,
    telemetry_app,
    train_app,
    transfer_app,
)
from forge.cli_v2_root import (
    register_v2_commands,
    setup_cli_logging_callback,
)

app = typer.Typer(
    name="forge",
    help="FORGE — VLA Model Distillation Pipeline v3",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.callback()(setup_cli_logging_callback)
register_v2_commands(app)

# Re-export app groups for backward compatibility and tests.
teacher_app = teacher_app
benchmark_app = benchmark_app
config_app = config_app
embodiment_app = embodiment_app
demo_app = demo_app
quantize_app = quantize_app
curriculum_app = curriculum_app
profile_app = profile_app
train_app = train_app
metrics_app = metrics_app
models_app = models_app
students_app = students_app
hyperparam_app = hyperparam_app
finetune_app = finetune_app
telemetry_app = telemetry_app
transfer_app = transfer_app
eval_app = eval_app


def main() -> None:
    """Console entry point with clean domain-error and Ctrl-C handling."""
    from forge.errors import ForgeError

    try:
        app()
    except ForgeError as exc:
        typer.echo(f"Error: {exc.message}", err=True)
        typer.echo(f"Hint: {exc.hint}", err=True)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        typer.echo("Interrupted. Partial outputs were left intact; rerun the same command to resume.", err=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
