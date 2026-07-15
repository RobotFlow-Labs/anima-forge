"""Action Head Factory — selects diffusion, flow, or chunking head via config."""

from forge.config import StudentConfig


def create_action_head(config: StudentConfig):
    """Factory for action prediction heads.

    Args:
        config: Student config with action_head_type field

    Returns:
        nn.Module — the action head

    Supported types:
        "diffusion" — v1 DDPM DiffusionActionHead (10 steps)
        "flow" — v2 FlowMatchingActionHead (1-4 steps)
        "chunk" — v2 ActionChunkHead (multi-step prediction)
    """
    head_type = config.action_head_type

    if head_type == "diffusion":
        from forge.modules.diffusion_head import DiffusionActionHead

        return DiffusionActionHead(
            d_model=config.bridge_d_model,
            d_action=config.action_dim,
            n_layers=config.action_head_layers,
            n_diffusion_steps=config.action_diffusion_steps,
        )
    elif head_type == "flow":
        from forge.modules.flow_head import FlowMatchingActionHead

        return FlowMatchingActionHead(
            d_model=config.bridge_d_model,
            d_action=config.action_dim,
            n_layers=config.action_head_layers,
            inference_steps=config.flow_inference_steps,
        )
    elif head_type == "chunk":
        from forge.modules.action_chunking import ActionChunkHead

        return ActionChunkHead(
            d_model=config.bridge_d_model,
            d_action=config.action_dim,
            horizon=config.action_horizon,
            n_layers=config.action_head_layers,
            chunk_overlap=config.chunk_overlap,
        )
    elif head_type == "consistency":
        from forge.modules.consistency_head import ConsistencyActionHead

        return ConsistencyActionHead(
            d_model=config.bridge_d_model,
            d_action=config.action_dim,
            n_layers=config.action_head_layers,
        )
    else:
        raise ValueError(
            f"Unknown action head type: {head_type}. Must be 'diffusion', 'flow', 'chunk', or 'consistency'."
        )
