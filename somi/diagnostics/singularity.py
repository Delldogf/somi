"""
SOMI 4.0 Singularity Detection (Robinson 2025)
=================================================

Embedding singularities: regions where the model's representation
collapses to a low-dimensional manifold, losing information.

Robinson et al. (2025) proved that standard transformers PRESERVE
singularities — once they form, they can't be resolved.

SOMI can RESOLVE singularities because:
1. Ricci-like flow (geometry equation) smooths curvature
2. Structural plasticity adds/removes connections
3. Mass-conductivity duality prevents degenerate regions

Detection algorithm (from Robinson 2025):
1. Variance proxy: find features with near-zero variance
2. Neighbor counting: count how many features are in the low-variance neighborhood
3. Fiber-bundle decomposition: separate into base (structural) and fiber (dynamic) components
"""

import torch
from typing import Dict, Tuple


def detect_singularities(
    phi: torch.Tensor,
    W: torch.Tensor,
    variance_threshold: float = 0.01,
    neighbor_radius: float = 0.1,
) -> Dict[str, float]:
    """
    Detect embedding singularities using Robinson's method.

    Args:
        phi: [batch, seq, hidden] or [N, hidden] embeddings
        W: [hidden, hidden] connectivity
        variance_threshold: Below this = potential singularity
        neighbor_radius: Radius for neighbor counting

    Returns:
        diagnostics: Dict with singularity metrics
    """
    diag = {}

    # Flatten to [N, hidden]
    if phi.dim() == 3:
        phi_flat = phi.detach().reshape(-1, phi.shape[-1])
    elif phi.dim() == 2:
        phi_flat = phi.detach()
    else:
        phi_flat = phi.detach().unsqueeze(0)

    H = phi_flat.shape[-1]

    # 1. Variance proxy — features with near-zero variance
    feature_var = phi_flat.var(dim=0)  # [hidden]
    singular_mask = feature_var < variance_threshold
    n_singular = singular_mask.sum().item()
    diag['singularity_count'] = n_singular
    diag['singularity_fraction'] = n_singular / H

    # 2. Severity — how small is the variance in singular regions?
    if n_singular > 0:
        singular_var = feature_var[singular_mask]
        diag['singularity_min_variance'] = singular_var.min().item()
        diag['singularity_mean_variance'] = singular_var.mean().item()
    else:
        diag['singularity_min_variance'] = feature_var.min().item()
        diag['singularity_mean_variance'] = feature_var.mean().item()

    # 3. Neighbor counting (Algorithm 1 from Robinson)
    # For each singular feature, count how many features are "close"
    # in the W-connectivity sense
    if n_singular > 0:
        singular_indices = torch.where(singular_mask)[0]
        avg_neighbors = 0.0
        for idx in singular_indices:
            # W[idx, :] tells us connectivity from this feature
            connected = W[idx.item(), :] > neighbor_radius
            avg_neighbors += connected.sum().item()
        avg_neighbors /= n_singular
        diag['singularity_avg_neighbors'] = avg_neighbors
    else:
        diag['singularity_avg_neighbors'] = 0.0

    # 4. Fiber-bundle decomposition (simplified)
    # Base space = low-frequency structure (first few eigenmodes)
    # Fiber = high-frequency dynamics
    # Singularity in fiber = local collapse; in base = global collapse
    L_sym = torch.eye(H, device=W.device) - W / (W.sum(dim=1, keepdim=True).clamp(min=1e-8))
    L_sym = 0.5 * (L_sym + L_sym.T)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)
        # Project phi onto low-frequency (first 5) and high-frequency modes
        n_base = min(5, H)
        phi_centered = phi_flat - phi_flat.mean(dim=0)
        base_proj = phi_centered @ eigenvectors[:, :n_base]  # Low freq
        fiber_proj = phi_centered @ eigenvectors[:, n_base:]  # High freq

        base_var = base_proj.var(dim=0).mean().item()
        fiber_var = fiber_proj.var(dim=0).mean().item() if eigenvectors.shape[1] > n_base else 0

        diag['singularity_base_variance'] = base_var
        diag['singularity_fiber_variance'] = fiber_var
        diag['singularity_in_base'] = base_var < variance_threshold
        diag['singularity_in_fiber'] = fiber_var < variance_threshold
    except Exception:
        diag['singularity_decomposition_failed'] = True

    # 5. Resolution capability
    # SOMI can resolve singularities via Ricci flow + structural plasticity
    # Metric: how much does the geometry "smooth" singular regions?
    if n_singular > 0:
        singular_W = W[singular_mask, :][:, singular_mask]
        diag['singularity_internal_connectivity'] = singular_W.abs().mean().item()
    else:
        diag['singularity_internal_connectivity'] = 0.0

    return diag
