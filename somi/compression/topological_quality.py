"""
SOMI 4.0 Topological Quality — Verify Compression Didn't Break Topology
=========================================================================

After compression, check that the network's topological properties
are preserved (connectivity, community structure, path lengths).
"""

import torch
from typing import Dict


def check_topological_quality(
    W_original: torch.Tensor,
    W_compressed: torch.Tensor,
) -> Dict[str, float]:
    """
    Compare topological properties before and after compression.

    Args:
        W_original: Original connectivity
        W_compressed: Compressed connectivity

    Returns:
        diagnostics: Quality metrics
    """
    diag = {}

    # Connection preservation
    orig_connections = (W_original.abs() > 1e-6).float()
    comp_connections = (W_compressed.abs() > 1e-6).float()
    preserved = (orig_connections * comp_connections).sum()
    total_orig = orig_connections.sum()
    diag['connection_preservation'] = (preserved / (total_orig + 1e-8)).item()

    # Weight correlation
    flat_orig = W_original.flatten()
    flat_comp = W_compressed.flatten()
    corr = torch.corrcoef(torch.stack([flat_orig, flat_comp]))[0, 1]
    diag['weight_correlation'] = corr.item() if not torch.isnan(corr) else 0.0

    # Spectral similarity (compare eigenvalues)
    try:
        L_orig = torch.eye(W_original.shape[0], device=W_original.device)
        L_orig -= W_original / (W_original.sum(1, keepdim=True).clamp(min=1e-8))
        L_comp = torch.eye(W_compressed.shape[0], device=W_compressed.device)
        L_comp -= W_compressed / (W_compressed.sum(1, keepdim=True).clamp(min=1e-8))

        eig_orig = torch.linalg.eigvalsh(0.5 * (L_orig + L_orig.T))
        eig_comp = torch.linalg.eigvalsh(0.5 * (L_comp + L_comp.T))

        spectral_diff = (eig_orig - eig_comp).abs().mean().item()
        diag['spectral_preservation'] = 1.0 / (1.0 + spectral_diff)
    except Exception:
        diag['spectral_preservation'] = -1.0

    # Row sum preservation
    row_diff = (W_original.sum(1) - W_compressed.sum(1)).abs().mean().item()
    diag['row_sum_preservation'] = 1.0 / (1.0 + row_diff)

    return diag
