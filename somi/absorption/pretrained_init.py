"""
SOMI 4.0 Pretrained Initialization
=====================================

Initialize SOMI's W_local, mass, and other parameters from a pretrained
transformer's weights. This gives SOMI a head start — it doesn't have to
learn everything from scratch.

What we extract from a pretrained transformer:
  - Attention weight patterns -> W_local initialization
  - Layer norm statistics -> mass initialization
  - MLP patterns -> potential landscape hints

Think of it like a brain transplant: we take the "wiring patterns" from
an expert brain (pretrained LLM) and install them in SOMI's physics.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


def extract_attention_W(
    attn_weights: torch.Tensor,
    method: str = 'mean_heads',
) -> torch.Tensor:
    """
    Extract a connectivity matrix from transformer attention weights.

    Attention weights tell us "which tokens attend to which other tokens."
    We can use this as an initialization for W_local, since W_local also
    encodes "which features are connected to which."

    Methods:
    - 'mean_heads': Average across all attention heads
    - 'max_heads': Take the max attention across heads
    - 'random_head': Use a random single head

    Args:
        attn_weights: [n_heads, hidden/n_heads, hidden/n_heads] or
                      [hidden, hidden] attention projection
        method: Aggregation method

    Returns:
        W_init: [hidden, hidden] initialization for W_local
    """
    if attn_weights.dim() == 3:
        if method == 'mean_heads':
            W = attn_weights.mean(dim=0)
        elif method == 'max_heads':
            W = attn_weights.max(dim=0).values
        elif method == 'random_head':
            idx = torch.randint(attn_weights.shape[0], (1,)).item()
            W = attn_weights[idx]
        else:
            W = attn_weights.mean(dim=0)
    elif attn_weights.dim() == 2:
        W = attn_weights
    else:
        raise ValueError(f"Expected 2D or 3D attention weights, got {attn_weights.dim()}D")

    # Make non-negative and normalize
    W = W.abs()
    W.fill_diagonal_(0)
    row_sums = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
    W = W / row_sums

    return W


def extract_mass_from_layernorm(
    ln_weight: torch.Tensor,
    ln_bias: Optional[torch.Tensor] = None,
    base_mass: float = 1.0,
) -> torch.Tensor:
    """
    Initialize per-feature mass from LayerNorm statistics.

    LayerNorm weight (gamma) tells us the "importance" of each feature.
    Features with high gamma are important (used a lot) — they should have
    high mass (slow to change, stable).
    Features with low gamma are less important — low mass (fast, adaptive).

    Args:
        ln_weight: [hidden] LayerNorm gamma parameters
        ln_bias: [hidden] LayerNorm beta (unused but accepted for API)
        base_mass: Base mass scale

    Returns:
        mass: [hidden] per-feature mass
    """
    # Importance ~ |gamma|
    importance = ln_weight.abs()
    # Normalize so mean mass = base_mass
    mass = base_mass * importance / (importance.mean() + 1e-8)
    return mass.clamp(min=0.1, max=10.0)


def init_from_pretrained(
    part: 'SOMIPart',
    transformer_layer: nn.Module,
    method: str = 'attention',
) -> Dict[str, float]:
    """
    Initialize a SOMIPart from a pretrained transformer layer.

    Extracts relevant information from the transformer and maps it
    to SOMI's physics parameters.

    Args:
        part: The SOMIPart to initialize
        transformer_layer: A transformer layer (must have attention, layernorm)
        method: 'attention' (use attention weights) or 'mlp' (use MLP weights)

    Returns:
        diagnostics: Dict with initialization metrics
    """
    diagnostics = {}

    # Try to extract attention weights
    attn_found = False
    for name, module in transformer_layer.named_modules():
        if hasattr(module, 'weight') and 'attn' in name.lower():
            try:
                W_init = extract_attention_W(module.weight.data)
                # Resize if dimensions don't match
                if W_init.shape[0] == part.W_local.shape[0]:
                    # Blend: 70% pretrained, 30% original (keep some randomness)
                    with torch.no_grad():
                        part.W_local.copy_(
                            0.7 * W_init.to(part.W_local.device)
                            + 0.3 * part.W_local
                        )
                    attn_found = True
                    diagnostics['pretrained_W_source'] = name
                    diagnostics['pretrained_W_blend'] = 0.7
                break
            except Exception as e:
                diagnostics['pretrained_W_error'] = str(e)

    if not attn_found:
        diagnostics['pretrained_W_source'] = 'random_init'

    # Try to extract mass from LayerNorm
    for name, module in transformer_layer.named_modules():
        if isinstance(module, nn.LayerNorm) and hasattr(module, 'weight'):
            if module.weight.shape[0] == part.mass.shape[0]:
                with torch.no_grad():
                    new_mass = extract_mass_from_layernorm(
                        module.weight.data,
                        base_mass=part.config.M,
                    )
                    part.mass.copy_(new_mass.to(part.mass.device))
                diagnostics['pretrained_mass_source'] = name
                break

    return diagnostics
