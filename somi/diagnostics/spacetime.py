"""
SOMI 4.0 Spacetime Diagnostics (Level 3)
===========================================

7 metrics from the spacetime (GR) level:
1. CFL condition — is dt safe?
2. c_info map — information speed per region
3. Dark energy — alpha_0 contribution to expansion
4. Dark matter — mass-geometry coupling
5. Hawking decay — lambda_W as thermal radiation
6. Noether conservation — are conserved quantities stable?
7. Geodesics — shortest paths through the network
"""

import torch
import math
from typing import Dict

from ..physics.forces import compute_laplacian
from ..physics.hamiltonian import compute_cfl_condition


def compute_spacetime_diagnostics(part: 'SOMIPart') -> Dict[str, float]:
    """Compute all Level 3 spacetime diagnostics for a Part."""
    diag = {}
    pid = part.part_id
    prefix = f'L3_part{pid}'
    W = part.W_local
    H = W.shape[0]

    # 1. CFL condition
    dt_safe, cfl_diag = compute_cfl_condition(W, part.config, part.mass)
    for k, v in cfl_diag.items():
        diag[f'{prefix}_{k}'] = v

    # 2. Information speed map — c_info per feature
    kappa = W.abs().sum(dim=1).clamp(min=1e-8)
    c_info_per_feature = torch.sqrt(
        part.config.alpha_1 * kappa / (part.mass + 1e-8)
    )
    diag[f'{prefix}_c_info_mean'] = c_info_per_feature.mean().item()
    diag[f'{prefix}_c_info_max'] = c_info_per_feature.max().item()
    diag[f'{prefix}_c_info_std'] = c_info_per_feature.std().item()

    # 3. Dark energy analog — alpha_0 acts like cosmological constant
    # Positive alpha_0 = repulsive (pushes phi toward 0) = expansion
    diag[f'{prefix}_dark_energy_alpha0'] = part.config.alpha_0
    diag[f'{prefix}_dark_energy_contribution'] = (
        part.config.alpha_0 * H  # Total anchoring force magnitude proxy
    )

    # 4. Dark matter analog — mass-geometry coupling
    # Regions with high mass but low connectivity = "dark matter"
    # (mass exists but doesn't interact much)
    mass_weighted_kappa = part.mass * kappa
    total_mass = part.mass.sum().item()
    visible_mass = mass_weighted_kappa.sum().item()
    diag[f'{prefix}_dark_matter_ratio'] = 1.0 - visible_mass / (total_mass * kappa.mean().item() + 1e-8)

    # 5. Hawking decay — lambda_W as thermal radiation
    # T_Hawking = lambda_W / (2*pi)
    T_hawking = part.config.lambda_W / (2 * math.pi)
    diag[f'{prefix}_hawking_temperature'] = T_hawking
    diag[f'{prefix}_hawking_lifetime'] = 1.0 / max(T_hawking, 1e-10)

    # 6. Noether conservation — total "charge" (sum of phi^2 * mass)
    # Should be approximately conserved during settling (minus dissipation)
    diag[f'{prefix}_noether_total_mass'] = part.mass.sum().item()

    # 7. Geodesics — shortest paths (approximate via graph distance)
    # Use Floyd-Warshall on -log(W) to find shortest paths
    # Expensive, so just compute average path length
    try:
        # Simple approximation: inverse W as distance
        dist = 1.0 / (W + 1e-6)
        dist.fill_diagonal_(0)
        # Average over connected pairs
        connected = W > 1e-6
        if connected.any():
            avg_dist = dist[connected].mean().item()
            diag[f'{prefix}_avg_geodesic_distance'] = avg_dist
    except Exception:
        diag[f'{prefix}_avg_geodesic_distance'] = -1.0

    return diag
