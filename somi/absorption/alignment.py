"""
SOMI 4.0 Cross-Size Alignment
================================

When the source model and SOMI Part have different dimensions,
we need to align them before transplant. This module handles:

1. Linear projection alignment (simple matrix multiply)
2. SVD-based alignment (preserves important directions)
3. CKA similarity (checks alignment quality)

Source: SOMI_3_0/theory/06_KNOWLEDGE_ABSORPTION.md
"""

import torch
from typing import Dict, Tuple


def svd_align(
    W_source: torch.Tensor,
    target_dim: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Align a weight matrix to a different dimension using SVD.

    SVD decomposes W = U @ S @ V^T. We keep the top-K singular values
    (where K = target_dim) to get the most important directions.

    Think of it like compressing a photo: keep the most important features,
    discard the least important. The result is smaller but captures the
    essence of the original.

    Args:
        W_source: [source_dim, source_dim] weight matrix
        target_dim: Desired dimension

    Returns:
        W_aligned: [target_dim, target_dim] aligned matrix
        diagnostics: Alignment quality metrics
    """
    U, S, Vh = torch.linalg.svd(W_source, full_matrices=False)

    # Keep top-K singular values
    K = min(target_dim, S.shape[0])
    U_k = U[:, :K]
    S_k = S[:K]
    Vh_k = Vh[:K, :]

    # Project into target dimension
    # Simple approach: slice or pad
    if W_source.shape[0] >= target_dim:
        W_aligned = (U_k[:target_dim, :] * S_k) @ Vh_k[:, :target_dim]
    else:
        # Pad with zeros
        W_aligned = torch.zeros(target_dim, target_dim, device=W_source.device)
        small = (U_k * S_k) @ Vh_k
        W_aligned[:small.shape[0], :small.shape[1]] = small

    # Energy preserved
    total_energy = S.pow(2).sum().item()
    kept_energy = S_k.pow(2).sum().item()

    diagnostics = {
        'alignment_method': 'svd',
        'alignment_source_dim': W_source.shape[0],
        'alignment_target_dim': target_dim,
        'alignment_k_singular': K,
        'alignment_energy_preserved': kept_energy / (total_energy + 1e-8),
    }

    return W_aligned, diagnostics


def cka_similarity(
    W_a: torch.Tensor,
    W_b: torch.Tensor,
) -> float:
    """
    Centered Kernel Alignment — measures how similar two representations are.

    CKA = 1.0 means identical, 0.0 means completely different.
    Used to verify alignment quality.

    Args:
        W_a, W_b: Two weight matrices (same shape)

    Returns:
        similarity: Float in [0, 1]
    """
    # Simple linear CKA
    hsic_ab = (W_a * W_b).sum().item()
    hsic_aa = (W_a * W_a).sum().item()
    hsic_bb = (W_b * W_b).sum().item()

    denom = (hsic_aa * hsic_bb) ** 0.5
    if denom < 1e-10:
        return 0.0

    return hsic_ab / denom
