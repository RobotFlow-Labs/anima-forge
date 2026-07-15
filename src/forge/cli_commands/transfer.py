"""Cross-embodiment transfer commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from forge.cli_commands.shared import emit_cli_error, emit_json

transfer_app = typer.Typer(name="transfer", help="Cross-embodiment transfer (PRD-30)")
console = Console()


@transfer_app.command("info")
def transfer_info(
    source: str = typer.Option("franka", help="Source embodiment"),
    target: str = typer.Option("ur5e", help="Target embodiment"),
    strategy: str = typer.Option("linear", help="Mapping strategy"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Show transfer mapping info between embodiments."""
    from forge.cross_embodiment import EmbodimentProfile, EmbodimentTransfer, TransferConfig
    from forge.embodiments.registry import EmbodimentRegistry

    registry = EmbodimentRegistry()
    try:
        src_profile = registry.get(source)
        tgt_profile = registry.get(target)
    except KeyError as exc:
        emit_cli_error(
            str(exc.args[0]),
            output_json=output_json,
            exit_code=2,
        )

    src_entry = EmbodimentProfile(
        name=src_profile.name,
        action_dim=src_profile.action_dim,
        joint_names=src_profile.joint_names,
        joint_min=src_profile.joint_min,
        joint_max=src_profile.joint_max,
        has_gripper=src_profile.has_gripper,
    )
    tgt_entry = EmbodimentProfile(
        name=tgt_profile.name,
        action_dim=tgt_profile.action_dim,
        joint_names=tgt_profile.joint_names,
        joint_min=tgt_profile.joint_min,
        joint_max=tgt_profile.joint_max,
        has_gripper=tgt_profile.has_gripper,
    )

    transfer = EmbodimentTransfer(src_entry, tgt_entry, TransferConfig(mapping_strategy=strategy))
    info = transfer.info()

    if output_json:
        emit_json(info)
    else:
        table = Table(title=f"Transfer: {source} → {target}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        for key, value in info.items():
            table.add_row(key, str(value))
        console.print(table)
