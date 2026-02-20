"""
SOMI 4.0 Stress-Guided Pruning
=================================

Remove connections with low stress (they're not doing useful work).
High-stress connections are the ones that matter for computation.
"""

import torch
from typing import Dict, Tuple


def stress_guided_pruning(
    W: torch.Tensor,
    S: torch.Tensor,
    prune_fraction: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Prune connections with lowest stress magnitude.

    Args:
        W: [hidden, hidden] connectivity
        S: [hidden, hidden] stress tensor
        prune_fraction: Fraction of connections to prune (0.3 = prune 30%)

    Returns:
        W_pruned: Pruned connectivity
        mask_new: Updated sparsity mask
        diagnostics: Pruning metrics
    """
    stress_mag = S.abs()
    # Only consider existing connections
    existing = W.abs() > 1e-6
    n_existing = existing.sum().item()
    n_to_prune = int(n_existing * prune_fraction)

    if n_to_prune == 0:
        mask_new = (W.abs() > 1e-6).float()
        return W, mask_new, {'pruned_count': 0}

    # Find lowest-stress connections among existing ones
    stress_existing = stress_mag * existing.float()
    flat = stress_existing.view(-1)
    existing_flat = existing.view(-1)

    existing_stress = flat[existing_flat]
    if len(existing_stress) > n_to_prune:
        threshold = existing_stress.topk(n_to_prune, largest=False).values[-1]
        prune_mask = (stress_mag <= threshold) & existing
    else:
        prune_mask = existing

    # Zero out pruned connections
    W_pruned = W.clone()
    W_pruned[prune_mask] = 0.0
    W_pruned.fill_diagonal_(0)
    mask_new = (W_pruned.abs() > 1e-6).float()

    diagnostics = {
        'pruned_count': prune_mask.sum().item(),
        'pruned_fraction': prune_mask.sum().item() / max(n_existing, 1),
        'remaining_connections': mask_new.sum().item(),
        'remaining_sparsity': 1.0 - mask_new.mean().item(),
    }

    return W_pruned, mask_new, diagnostics
