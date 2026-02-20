"""
SOMI Checkpoint — Variable-Size Save/Load with Lifetime Metadata
===================================================================

One model forever needs one checkpoint format that handles:
  - Variable hidden_dim (model grew since last save)
  - Variable n_parts (Parts were added)
  - Variable n_systems (Systems were added)
  - Variable white_matter_rank (changed after recalibration)
  - Absorption metadata (who was absorbed, when, what was gained)
  - Compression metadata (when, what was pruned)
  - Growth metadata (when, from what size to what size)

The checkpoint is a dict:
  {
    'config': SOMIBrainConfig (serialized),
    'brain_state': brain.state_dict(),
    'metadata': {
        'model_id': str,
        'created_at': str,
        'step': int,
        'history': [...],  # list of events
    }
  }

Usage:
    save_checkpoint(brain, path, step=1000)
    brain, meta = load_checkpoint(path, input_dim=768, output_dim=32000)
"""

import torch
import datetime
import uuid
from typing import Dict, Optional, Tuple
from dataclasses import asdict

from .config import SOMIBrainConfig
from .brain.circuit_brain import SOMICircuitBrain


def save_checkpoint(
    brain: SOMICircuitBrain,
    path: str,
    step: int = 0,
    model_id: Optional[str] = None,
    extra_metadata: Optional[Dict] = None,
    history: Optional[list] = None,
) -> str:
    """
    Save a SOMI brain checkpoint with full metadata.

    Args:
        brain: SOMICircuitBrain to save
        path: File path (e.g., 'checkpoints/somi_v1.pt')
        step: Current training step
        model_id: Unique model ID (auto-generated if None)
        extra_metadata: Any extra info to store
        history: List of lifetime events (grow, compress, absorb)

    Returns:
        model_id: The model's unique ID
    """
    if model_id is None:
        model_id = str(uuid.uuid4())[:8]

    config_dict = {}
    for k, v in asdict(brain.config).items():
        if isinstance(v, (int, float, str, bool, list, type(None))):
            config_dict[k] = v
        elif isinstance(v, tuple):
            config_dict[k] = list(v)

    checkpoint = {
        'config': config_dict,
        'brain_state': brain.state_dict(),
        'input_dim': brain.x_encoder.in_features,
        'output_dim': brain.y_decoder.out_features,
        'metadata': {
            'model_id': model_id,
            'created_at': datetime.datetime.now().isoformat(),
            'step': step,
            'hidden_dim': brain.config.hidden_dim,
            'n_parts': brain.config.n_parts,
            'n_systems': len(brain.systems),
            'history': history or [],
            'extra': extra_metadata or {},
        },
    }

    torch.save(checkpoint, path)
    return model_id


def load_checkpoint(
    path: str,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    device: str = 'cpu',
) -> Tuple[SOMICircuitBrain, Dict]:
    """
    Load a SOMI brain from checkpoint, handling variable sizes.

    Args:
        path: Path to checkpoint file
        input_dim: Override input dim (None = use saved)
        output_dim: Override output dim (None = use saved)
        device: Device to load onto

    Returns:
        brain: Reconstructed SOMICircuitBrain
        metadata: Lifetime metadata dict
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config_dict = checkpoint['config']
    if 'system_routes' in config_dict and isinstance(config_dict['system_routes'], list):
        config_dict['system_routes'] = [
            list(r) for r in config_dict['system_routes']
        ]
    for k, v in config_dict.items():
        if isinstance(v, list) and k in ('delay_range', 'alpha_schedule'):
            config_dict[k] = tuple(v)

    config = SOMIBrainConfig(**{
        k: v for k, v in config_dict.items()
        if k in SOMIBrainConfig.__dataclass_fields__
    })

    saved_input_dim = checkpoint.get('input_dim', 768)
    saved_output_dim = checkpoint.get('output_dim', 32000)

    brain = SOMICircuitBrain(
        config,
        input_dim=input_dim or saved_input_dim,
        output_dim=output_dim or saved_output_dim,
    )

    # Load state dict with size tolerance
    saved_state = checkpoint['brain_state']
    current_state = brain.state_dict()

    compatible_state = {}
    for key in current_state:
        if key in saved_state:
            if saved_state[key].shape == current_state[key].shape:
                compatible_state[key] = saved_state[key]
            else:
                compatible_state[key] = current_state[key]
                src = saved_state[key]
                dst = compatible_state[key]
                slices = tuple(
                    slice(0, min(s, d))
                    for s, d in zip(src.shape, dst.shape)
                )
                dst_slices = tuple(
                    slice(0, min(s, d))
                    for s, d in zip(src.shape, dst.shape)
                )
                dst[dst_slices] = src[slices]
        else:
            compatible_state[key] = current_state[key]

    brain.load_state_dict(compatible_state, strict=False)
    brain = brain.to(device)

    metadata = checkpoint.get('metadata', {})
    return brain, metadata


def record_event(
    history: list,
    event_type: str,
    step: int,
    details: Optional[Dict] = None,
):
    """
    Record a lifetime event (grow, compress, absorb, etc.).

    Args:
        history: The history list to append to
        event_type: 'grow', 'compress', 'absorb', 'train', etc.
        step: Training step when event occurred
        details: Additional event-specific info
    """
    history.append({
        'type': event_type,
        'step': step,
        'timestamp': datetime.datetime.now().isoformat(),
        'details': details or {},
    })
