"""
SOMI 4.0 Adaptive Compression Pipeline
=========================================

Unified pipeline that applies all compression methods in the right order
and verifies quality after each step.
"""

import torch
from typing import Dict

from .mass_precision import mass_guided_quantization
from .stress_pruning import stress_guided_pruning
from .spectral_rank import spectral_rank_selection
from .topological_quality import check_topological_quality


def adaptive_compress(
    part: 'SOMIPart',
    target_compression: float = 2.0,
    quality_threshold: float = 0.9,
) -> Dict[str, float]:
    """
    Adaptively compress a Part using physics guidance.

    Steps:
    1. Spectral rank selection (how many eigenmodes do we need?)
    2. Stress-guided pruning (remove unneeded connections)
    3. Mass-guided quantization (lower precision for light features)
    4. Quality check (did we break anything?)

    Args:
        part: SOMIPart to compress
        target_compression: Target compression ratio (2.0 = half the memory)
        quality_threshold: Minimum quality to accept (0.9 = 90%)

    Returns:
        diagnostics: All compression metrics
    """
    all_diag = {}
    W_original = part.W_local.clone()

    # 1. Spectral rank selection
    from ..physics.settling import compute_eigendecomposition
    from ..physics.forces import compute_laplacian
    L_rw = compute_laplacian(part.W_local)
    eigenvalues, _, eigen_diag = compute_eigendecomposition(L_rw)
    rank, rank_diag = spectral_rank_selection(eigenvalues, 0.95)
    all_diag.update(rank_diag)

    # 2. Stress-guided pruning
    from ..physics.geometry import compute_stress_tensor
    # Use a dummy phi for stress computation
    H = part.config.hidden_dim
    phi_dummy = torch.randn(1, 1, H, device=part.W_local.device) * 0.1
    S, _ = compute_stress_tensor(phi_dummy, phi_dummy, part.config)

    prune_fraction = min(0.5, 1.0 - 1.0 / target_compression)
    W_pruned, mask_new, prune_diag = stress_guided_pruning(
        part.W_local, S, prune_fraction
    )
    all_diag.update(prune_diag)

    # 3. Mass-guided quantization
    W_quantized, quant_diag = mass_guided_quantization(
        W_pruned, part.mass, high_precision_threshold=0.7
    )
    all_diag.update(quant_diag)

    # 4. Quality check
    quality = check_topological_quality(W_original, W_quantized)
    all_diag.update(quality)

    # Apply if quality is acceptable
    overall_quality = quality.get('weight_correlation', 0)
    if overall_quality >= quality_threshold:
        with torch.no_grad():
            part.W_local.copy_(W_quantized)
            part.mask.copy_(mask_new)
        all_diag['compression_applied'] = True
    else:
        all_diag['compression_applied'] = False
        all_diag['compression_rejected_quality'] = overall_quality

    return all_diag
