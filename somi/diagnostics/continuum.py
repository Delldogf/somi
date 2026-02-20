"""
SOMI 4.0 Continuum Diagnostics (Level 2)
===========================================

7 metrics from the spatial continuum limit:
1. Mass-conductivity duality — rho * kappa ≈ 1/alpha_1
2. Spectral decomposition — eigenmode energy distribution
3. Scaling laws — does alpha_1 scale correctly with N?
4. Weyl's law — eigenvalue count vs. frequency
5. Analytical eigenvalues — comparison to continuum prediction
6. Ricci flow — geometry evolution direction
7. Coordination locality — is coupling truly local?
"""

import torch
import math
from typing import Dict

from ..physics.forces import compute_laplacian
from ..physics.settling import compute_eigendecomposition
from ..physics.geometry import mass_conductivity_constraint


def compute_continuum_diagnostics(part: 'SOMIPart') -> Dict[str, float]:
    """Compute all Level 2 continuum diagnostics for a Part."""
    diag = {}
    pid = part.part_id
    prefix = f'L2_part{pid}'
    W = part.W_local
    H = W.shape[0]

    # 1. Mass-conductivity duality
    _, mc_diag = mass_conductivity_constraint(
        part.mass, W, part.config.alpha_1
    )
    diag[f'{prefix}_mc_violation'] = mc_diag['mass_conductivity_violation']
    diag[f'{prefix}_mc_kappa_mean'] = mc_diag['kappa_mean']

    # 2. Spectral decomposition — energy in each eigenmode
    L_rw = compute_laplacian(W)
    eigenvalues, _, _ = compute_eigendecomposition(L_rw)
    pos_eigs = eigenvalues[eigenvalues > 1e-6]

    if len(pos_eigs) > 0:
        # Mode energy distribution
        eig_energy = pos_eigs / pos_eigs.sum()
        entropy = -(eig_energy * (eig_energy + 1e-10).log()).sum().item()
        diag[f'{prefix}_spectral_entropy'] = entropy
        diag[f'{prefix}_spectral_effective_dim'] = math.exp(entropy)

        # Top mode concentration
        diag[f'{prefix}_top1_energy'] = pos_eigs.max().item() / pos_eigs.sum().item()
        diag[f'{prefix}_top5_energy'] = (
            pos_eigs.topk(min(5, len(pos_eigs))).values.sum().item()
            / pos_eigs.sum().item()
        )

    # 3. Scaling law check
    expected_alpha = 1.0 * (128.0 / H)  # alpha_1 ~ N^(-2/d), d=1
    actual_alpha = part.config.alpha_1
    diag[f'{prefix}_scaling_expected_alpha'] = expected_alpha
    diag[f'{prefix}_scaling_actual_alpha'] = actual_alpha
    diag[f'{prefix}_scaling_ratio'] = actual_alpha / max(expected_alpha, 1e-8)

    # 4. Weyl's law — N(lambda) ~ C * lambda^(d/2)
    # For 1D: N(lambda) should grow linearly with lambda
    if len(pos_eigs) > 5:
        mid = len(pos_eigs) // 2
        ratio = mid / (pos_eigs[mid].item() + 1e-8)
        diag[f'{prefix}_weyl_ratio'] = ratio

    # 5. Coordination locality check
    # How "local" is W? Measure how fast W_ij decays with |i-j|
    if H > 10:
        local_weight = 0.0
        total_weight = W.abs().sum().item()
        for offset in range(1, min(5, H)):
            band = torch.diagonal(W, offset).abs().sum().item()
            band += torch.diagonal(W, -offset).abs().sum().item()
            local_weight += band
        diag[f'{prefix}_locality_fraction'] = local_weight / max(total_weight, 1e-8)

    return diag
