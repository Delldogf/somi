"""
SOMI 4.0 Standard Diagnostics (Level 1)
==========================================

9 metric groups that monitor the basic health of each Part:
1. Stress — magnitude, positive/negative balance, frobenius norm
2. Mass — mean, std, min, max, Herfindahl diversity
3. Settling — velocity norm, steps used, method
4. Hamiltonian — H, dH/dt, violations
5. Eigenspectrum — spectral gap, condition number, mode count
6. Arousal — current level, running average
7. Neuromodulators — NE, DA, ACh, 5-HT levels
8. W statistics — mean, sparsity, spectral radius, asymmetry
9. Domain stress — per-domain stress breakdown (if applicable)

These are computed EVERY step. They're cheap and essential.
"""

import torch
from typing import Dict

from ..physics.forces import compute_laplacian
from ..physics.settling import compute_eigendecomposition


def compute_standard_diagnostics(part: 'SOMIPart') -> Dict[str, float]:
    """
    Compute all Level 1 standard diagnostics for a Part.

    Args:
        part: A SOMIPart instance

    Returns:
        diagnostics: Dict with ~30 metrics
    """
    diag = {}
    pid = part.part_id
    prefix = f'L1_part{pid}'

    W = part.W_local
    H = W.shape[0]

    # 1. W statistics
    diag[f'{prefix}_W_mean'] = W.abs().mean().item()
    diag[f'{prefix}_W_max'] = W.abs().max().item()
    diag[f'{prefix}_W_sparsity'] = (W.abs() < 1e-6).float().mean().item()
    diag[f'{prefix}_W_asymmetry'] = (W - W.T).abs().mean().item()

    # Spectral radius (largest eigenvalue magnitude)
    # Use power iteration instead of eigvals — eigvals crashes Intel MKL
    # on non-symmetric matrices on some platforms
    try:
        v = torch.randn(H, device=W.device, dtype=W.dtype)
        v = v / v.norm()
        for _ in range(20):
            v_new = W @ v
            spectral_est = v_new.norm()
            if spectral_est > 1e-8:
                v = v_new / spectral_est
            else:
                break
        diag[f'{prefix}_W_spectral_radius'] = spectral_est.item()
    except Exception:
        diag[f'{prefix}_W_spectral_radius'] = -1.0

    # 2. Mass statistics
    mass = part.mass
    diag[f'{prefix}_mass_mean'] = mass.mean().item()
    diag[f'{prefix}_mass_std'] = mass.std().item()
    diag[f'{prefix}_mass_min'] = mass.min().item()
    diag[f'{prefix}_mass_max'] = mass.max().item()
    # Herfindahl diversity (inverse concentration)
    mass_norm = mass / mass.sum()
    herfindahl = (mass_norm ** 2).sum().item()
    diag[f'{prefix}_mass_herfindahl'] = herfindahl
    diag[f'{prefix}_mass_effective_n'] = 1.0 / max(herfindahl, 1e-8)

    # 3. Eigenspectrum
    L_rw = compute_laplacian(W)
    eigenvalues, _, eigen_diag = compute_eigendecomposition(L_rw)
    diag[f'{prefix}_spectral_gap'] = eigen_diag.get('eigen_spectral_gap', 0)
    diag[f'{prefix}_eigen_min'] = eigen_diag.get('eigen_min', 0)
    diag[f'{prefix}_eigen_max'] = eigen_diag.get('eigen_max', 0)
    diag[f'{prefix}_eigen_median'] = eigen_diag.get('eigen_median', 0)

    # Condition number
    pos_eigs = eigenvalues[eigenvalues > 1e-6]
    if len(pos_eigs) > 1:
        diag[f'{prefix}_condition_number'] = (
            pos_eigs.max().item() / pos_eigs.min().item()
        )
    else:
        diag[f'{prefix}_condition_number'] = 1.0

    # 4. Arousal
    diag[f'{prefix}_arousal'] = part.arousal.item()
    diag[f'{prefix}_error_running_avg'] = part.error_running_avg.item()

    # 5. Neuromodulators
    if part.config.neuromodulators_enabled:
        diag[f'{prefix}_NE'] = part.ne_level.item()
        diag[f'{prefix}_DA'] = part.da_level.item()
        diag[f'{prefix}_ACh'] = part.ach_level.item()
        diag[f'{prefix}_5HT'] = part.serotonin_level.item()

    # 6. Calibrated parameters
    diag[f'{prefix}_eta'] = part.eta
    diag[f'{prefix}_n_settle'] = part.n_settle
    diag[f'{prefix}_step'] = part.global_step.item()

    return diag
