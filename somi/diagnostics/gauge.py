"""
SOMI 4.0 Gauge Diagnostics (Level 4)
=======================================

6 metrics from gauge theory:
1. Wilson loops — curvature around closed paths
2. Discrete curvature — per-edge curvature
3. Chern-Simons invariant — topological characteristic
4. Instantons — sudden topology changes
5. Yang-Mills action — total gauge field energy
6. Gauge regularization — connection smoothness
"""

import torch
import math
from typing import Dict


def compute_gauge_diagnostics(
    brain: 'SOMICircuitBrain',
) -> Dict[str, float]:
    """Compute all Level 4 gauge diagnostics for the brain."""
    diag = {}
    prefix = 'L4'

    # 1. Wilson loops for each System's route
    for sys_id, route in enumerate(brain.config.system_routes):
        if len(route) >= 3:
            loop = route + [route[0]]  # Close the loop
            _, wilson_diag = brain.white_matter.compute_wilson_loop(loop)
            for k, v in wilson_diag.items():
                diag[f'{prefix}_sys{sys_id}_{k}'] = v

    # 2. Tract-level diagnostics
    tract_diag = brain.white_matter.get_all_curvatures()
    for k, v in tract_diag.items():
        diag[f'{prefix}_{k}'] = v

    # 3. Yang-Mills action — total "curvature energy" of gauge field
    # S_YM = sum of tract_norm^2
    total_ym = sum(
        v ** 2 for k, v in tract_diag.items() if k.endswith('_norm')
    )
    diag[f'{prefix}_yang_mills_action'] = total_ym

    # 4. Gauge smoothness — variance of tract norms
    norms = [v for k, v in tract_diag.items() if k.endswith('_norm')]
    if norms:
        import statistics
        diag[f'{prefix}_gauge_smoothness'] = statistics.stdev(norms) if len(norms) > 1 else 0.0
        diag[f'{prefix}_gauge_mean_norm'] = statistics.mean(norms)

    # 5. Chern-Simons invariant (simplified)
    # For discrete graphs: CS = sum_triangles(trace(W_ab @ W_bc @ W_ca))
    # We approximate using Wilson loop traces
    cs_total = 0.0
    cs_count = 0
    for sys_id, route in enumerate(brain.config.system_routes):
        if len(route) >= 3:
            loop = route + [route[0]]
            W_loop, _ = brain.white_matter.compute_wilson_loop(loop)
            cs_total += W_loop.trace().item()
            cs_count += 1
    if cs_count > 0:
        diag[f'{prefix}_chern_simons'] = cs_total / cs_count
    else:
        diag[f'{prefix}_chern_simons'] = 0.0

    return diag
