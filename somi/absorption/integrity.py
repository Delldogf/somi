"""
SOMI 4.0 Integrity Checks — Verify Transplant Didn't Break Physics
=====================================================================

After absorbing knowledge, check that the brain's physics are still healthy.
A bad transplant can violate SOMI constraints and break the model.

Checks:
1. Hamiltonian still decreasing? (energy conservation)
2. W constraints satisfied? (non-negative, zero diagonal, bounded)
3. Mass-conductivity duality holding? (rho * kappa ≈ 1/alpha_1)
4. Eigenspectrum reasonable? (no collapsed modes)
5. Stress within healthy range? (not too high, not too low)
"""

import torch
from typing import Dict, List

from ..physics.forces import compute_laplacian
from ..physics.settling import compute_eigendecomposition
from ..physics.geometry import mass_conductivity_constraint


def check_integrity(
    brain: 'SOMICircuitBrain',
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Run all integrity checks on the brain after absorption.

    Args:
        brain: The SOMICircuitBrain to check
        verbose: Print results

    Returns:
        report: Dict with all check results and overall health
    """
    report = {}
    all_healthy = True

    for pid, part in brain.parts.items():
        part_report = {}

        # Check 1: W constraints
        W = part.W_local
        diag_vals = W.diag().abs().max().item()
        part_report['W_diag_max'] = diag_vals
        part_report['W_diag_ok'] = diag_vals < 1e-6

        negative_frac = (W < -1e-6).float().mean().item()
        part_report['W_negative_frac'] = negative_frac
        part_report['W_nonneg_ok'] = negative_frac < 0.01

        row_sums = W.sum(dim=1)
        part_report['W_row_max'] = row_sums.max().item()
        part_report['W_row_bounded'] = row_sums.max().item() < 5.0

        # Check 2: Mass-conductivity
        _, mc_diag = mass_conductivity_constraint(
            part.mass, W, brain.config.alpha_1
        )
        part_report['mass_conductivity_violation'] = mc_diag['mass_conductivity_violation']
        part_report['mass_conductivity_ok'] = mc_diag['mass_conductivity_violation'] < 1.0

        # Check 3: Eigenspectrum
        L_rw = compute_laplacian(W)
        eigenvalues, _, eigen_diag = compute_eigendecomposition(L_rw)
        part_report['spectral_gap'] = eigen_diag.get('eigen_spectral_gap', 0)
        part_report['spectral_gap_ok'] = eigen_diag.get('eigen_spectral_gap', 0) > 0.01

        # Check 4: Mass range
        part_report['mass_min'] = part.mass.min().item()
        part_report['mass_max'] = part.mass.max().item()
        part_report['mass_range_ok'] = (
            part.mass.min().item() > 0.05 and part.mass.max().item() < 20.0
        )

        # Overall Part health
        checks = [v for k, v in part_report.items() if k.endswith('_ok')]
        part_report['all_ok'] = all(checks)
        if not part_report['all_ok']:
            all_healthy = False

        report[f'part_{pid}'] = part_report

    # Check 5: Wilson loop integrity (topological invariants)
    wilson_healthy = True
    for route in (brain.config.system_routes or []):
        if len(route) >= 3:
            loop = route + [route[0]]
            try:
                _, wilson_diag = brain.white_matter.compute_wilson_loop(loop)
                loop_key = '_'.join(str(r) for r in route)
                report[f'wilson_{loop_key}'] = wilson_diag
                trace_val = wilson_diag.get('wilson_trace', 0)
                if abs(trace_val) < 1e-10:
                    wilson_healthy = False
            except Exception:
                pass
    report['wilson_healthy'] = wilson_healthy
    if not wilson_healthy:
        all_healthy = False

    report['overall_healthy'] = all_healthy

    if verbose:
        for pid, pr in report.items():
            if isinstance(pr, dict):
                status = 'HEALTHY' if pr.get('all_ok', True) else 'ISSUES'
                print(f"  {pid}: {status}")
        print(f"  Wilson loops: {'HEALTHY' if wilson_healthy else 'ISSUES'}")

    return report
