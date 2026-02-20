"""
SOMI 4.0 Spectral Rank Selection
===================================

Use eigenvalue energy distribution to determine the minimum number
of eigenmodes needed. Modes with negligible energy can be discarded.
"""

import torch
from typing import Dict, Tuple


def spectral_rank_selection(
    eigenvalues: torch.Tensor,
    energy_threshold: float = 0.95,
) -> Tuple[int, Dict[str, float]]:
    """
    Determine minimum eigenmode count preserving given energy fraction.

    Args:
        eigenvalues: [n_modes] sorted eigenvalues
        energy_threshold: Fraction of energy to preserve (0.95 = keep 95%)

    Returns:
        optimal_rank: Minimum modes needed
        diagnostics: Spectral rank metrics
    """
    pos = eigenvalues[eigenvalues > 1e-8]
    if len(pos) == 0:
        return 1, {'spectral_rank': 1, 'total_energy': 0.0}

    total_energy = pos.sum().item()
    cumulative = torch.cumsum(pos, dim=0)
    cumulative_fraction = cumulative / total_energy

    # Find minimum rank that exceeds threshold
    above_threshold = cumulative_fraction >= energy_threshold
    if above_threshold.any():
        optimal_rank = above_threshold.nonzero()[0].item() + 1
    else:
        optimal_rank = len(pos)

    diagnostics = {
        'spectral_rank': optimal_rank,
        'spectral_total_modes': len(pos),
        'spectral_compression_ratio': len(pos) / max(optimal_rank, 1),
        'spectral_energy_preserved': (
            cumulative[min(optimal_rank - 1, len(cumulative) - 1)].item() / total_energy
        ),
        'spectral_total_energy': total_energy,
    }

    return optimal_rank, diagnostics
