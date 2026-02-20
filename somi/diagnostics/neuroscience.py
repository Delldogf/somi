"""
SOMI 4.0 Neuroscience Diagnostics
====================================

13 brain-inspired diagnostic tests:

Core 10 (from neuroscience flywheel):
1.  Eigenmode-Task Alignment (Raj) — do eigenmodes match task structure?
2.  Structure-Function Coupling (Tobyne) — does W predict activity?
3.  Mass Spectrum Diversity (Paquola) — is there a healthy mass hierarchy?
4.  Oscillatory Power Spectrum — frequency distribution of dynamics
5.  E/I Balance — excitation/inhibition ratio
6.  Network Modularity — does W form clusters?
7.  Criticality Index — is the system at the edge of chaos?
8.  Hebbian Consistency — does W follow Hebb's rule?
9.  Hamiltonian Health — is energy decreasing properly?
10. Small-World Properties — clustering + short paths?

Paper Extensions (3):
11. Generalization (Tobyne extension) — R-squared + complexity
12. Receiver ID (Paquola) — can we identify features by their mass?
13. Balanced Output (Paquola) — is output balanced across features?

Source: SOMI_3_0/theory/10_NEUROSCIENCE_FLYWHEEL.md
"""

import torch
import math
from typing import Dict

from ..physics.forces import compute_laplacian
from ..physics.settling import compute_eigendecomposition


def compute_neuroscience_diagnostics(part: 'SOMIPart') -> Dict[str, float]:
    """Compute all 13 neuroscience diagnostics for a Part."""
    diag = {}
    pid = part.part_id
    prefix = f'neuro_part{pid}'
    W = part.W_local
    H = W.shape[0]

    # 1. Eigenmode-Task Alignment
    L_rw = compute_laplacian(W)
    eigenvalues, eigenvectors, _ = compute_eigendecomposition(L_rw)
    # Use mass as proxy for "task relevance"
    mass_proj = eigenvectors.T @ part.mass  # Project mass onto eigenmodes
    alignment = (mass_proj ** 2).sum().item() / (part.mass.norm().item() ** 2 + 1e-8)
    diag[f'{prefix}_eigenmode_task_alignment'] = alignment

    # 2. Structure-Function Coupling
    # Correlation between W_ij and phi_i * phi_j (if available)
    # Proxy: correlation between row sums and mass
    row_sums = W.sum(dim=1)
    sf_corr = torch.corrcoef(torch.stack([row_sums, part.mass]))[0, 1].item()
    diag[f'{prefix}_structure_function_coupling'] = sf_corr if not math.isnan(sf_corr) else 0.0

    # 3. Mass Spectrum Diversity
    mass_sorted = part.mass.sort().values
    mass_range = mass_sorted[-1].item() - mass_sorted[0].item()
    mass_cv = part.mass.std().item() / (part.mass.mean().item() + 1e-8)
    diag[f'{prefix}_mass_diversity'] = mass_cv
    diag[f'{prefix}_mass_range'] = mass_range

    # 4. E/I Balance
    if part.config.dales_law and part.ei_mask is not None:
        n_exc = (part.ei_mask > 0).sum().item()
        n_inh = (part.ei_mask < 0).sum().item()
        exc_weight = W[:, part.ei_mask > 0].abs().sum().item() if n_exc > 0 else 0
        inh_weight = W[:, part.ei_mask < 0].abs().sum().item() if n_inh > 0 else 0
        total = exc_weight + inh_weight + 1e-8
        diag[f'{prefix}_ei_balance'] = exc_weight / total
        diag[f'{prefix}_ei_ratio'] = exc_weight / (inh_weight + 1e-8)
    else:
        # Without Dale's Law, measure positive vs negative weights
        diag[f'{prefix}_ei_balance'] = (W > 0).float().mean().item()

    # 5. Network Modularity (simplified Newman modularity)
    # Q = (1/2m) * sum_ij (A_ij - k_i*k_j/2m) * delta(c_i, c_j)
    # Simplified: use spectral gap as modularity proxy
    pos_eigs = eigenvalues[eigenvalues > 1e-6]
    if len(pos_eigs) > 1:
        diag[f'{prefix}_modularity_proxy'] = (
            pos_eigs[1].item() - pos_eigs[0].item()
        )
    else:
        diag[f'{prefix}_modularity_proxy'] = 0.0

    # 6. Criticality Index (edge of chaos)
    # System is critical when spectral radius ≈ 1
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
        spectral_radius = spectral_est.item()
        diag[f'{prefix}_criticality_index'] = abs(spectral_radius - 1.0)
        diag[f'{prefix}_spectral_radius'] = spectral_radius
        diag[f'{prefix}_at_criticality'] = abs(spectral_radius - 1.0) < 0.1
    except Exception:
        diag[f'{prefix}_criticality_index'] = -1.0

    # 7. Hebbian Consistency
    # Does W_ij correlate with phi_i * phi_j?
    # Use mass as proxy for average activity
    outer_mass = part.mass.unsqueeze(1) * part.mass.unsqueeze(0)
    # Correlation between W and outer product of mass
    hebbian_score = torch.corrcoef(
        torch.stack([W.flatten(), outer_mass.flatten()])
    )[0, 1].item()
    diag[f'{prefix}_hebbian_consistency'] = hebbian_score if not math.isnan(hebbian_score) else 0.0

    # 8. Small-World Properties
    # Clustering coefficient (average local clustering)
    # C = fraction of triangles / possible triangles
    A = (W > 1e-6).float()  # Binary adjacency
    A.fill_diagonal_(0)
    triangles = torch.trace(A @ A @ A) / 6.0
    possible = A.sum() * (A.sum() - 1) / 2
    clustering = triangles.item() / max(possible.item(), 1)
    diag[f'{prefix}_clustering_coefficient'] = clustering

    # Average path length (approximate via BFS depth)
    avg_degree = A.sum(dim=1).mean().item()
    if avg_degree > 0:
        diag[f'{prefix}_avg_degree'] = avg_degree
        # Small-world: high clustering + short paths
        diag[f'{prefix}_small_world_index'] = (
            clustering / max(avg_degree / H, 1e-8)
        )

    # 9. Hamiltonian Health
    diag[f'{prefix}_hamiltonian_violations'] = part.hamiltonian_tracker.violations
    diag[f'{prefix}_hamiltonian_violation_rate'] = (
        part.hamiltonian_tracker.violations / max(part.hamiltonian_tracker.total_steps, 1)
    )

    # 10. Oscillatory Power Spectrum (from eigenvalues)
    if len(pos_eigs) > 0:
        # Frequencies = sqrt(eigenvalues)
        freqs = torch.sqrt(pos_eigs.clamp(min=1e-8))
        diag[f'{prefix}_freq_min'] = freqs.min().item()
        diag[f'{prefix}_freq_max'] = freqs.max().item()
        diag[f'{prefix}_freq_range'] = freqs.max().item() - freqs.min().item()

    # 11-13. Paper extensions
    # 11. Generalization (R-squared between structure and function)
    diag[f'{prefix}_generalization_r2'] = sf_corr ** 2 if not math.isnan(sf_corr) else 0.0

    # 12. Receiver ID — can features be uniquely identified by mass?
    mass_unique = len(torch.unique(part.mass.round(decimals=2)))
    diag[f'{prefix}_receiver_id_unique'] = mass_unique
    diag[f'{prefix}_receiver_id_fraction'] = mass_unique / H

    # 13. Balanced Output — is activity spread or concentrated?
    mass_entropy = -(
        (part.mass / part.mass.sum()) * (part.mass / part.mass.sum() + 1e-10).log()
    ).sum().item()
    diag[f'{prefix}_balanced_output_entropy'] = mass_entropy
    diag[f'{prefix}_balanced_output_max_entropy'] = math.log(H)

    return diag
