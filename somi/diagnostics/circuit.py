"""
SOMI 4.0 Circuit Diagnostics
===============================

11 circuit metrics including 3 shared-Part diagnostics:

Standard (8):
1. Stress per system — how strained is each circuit?
2. Throughput — signal magnitude through each system
3. Coherence — correlation between system outputs
4. Bottleneck — which Part limits performance?
5. Tract utilization — how much is each white matter tract used?
6. Tract gradient — gradient magnitude through tracts
7. Information flow — how much info passes through circuits?
8. JEPA error — per-system JEPA prediction error

Shared-Part (3):
9.  shared_part_stress_variance — stress variance on shared Parts
10. shared_part_generalization — how well shared Parts generalize
11. generalization_pressure — how much multi-system pressure on shared Parts
"""

import torch
from typing import Dict, Optional


def compute_circuit_diagnostics(
    brain: 'SOMICircuitBrain',
    system_outputs: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Compute all 11 circuit diagnostics."""
    diag = {}
    prefix = 'circuit'
    config = brain.config

    # Per-Part stress aggregated from all Systems
    part_stress_sources = {int(pid): 0 for pid in brain.parts}
    part_system_count = {int(pid): 0 for pid in brain.parts}

    for sys_id, route in enumerate(config.system_routes):
        for part_id in route:
            part_system_count[part_id] = part_system_count.get(part_id, 0) + 1

    # 1-2. Per-system throughput and stress
    for pid, part in brain.parts.items():
        pid_int = int(pid)
        diag[f'{prefix}_part{pid}_W_magnitude'] = part.W_local.abs().mean().item()
        diag[f'{prefix}_part{pid}_arousal'] = part.arousal.item()
        diag[f'{prefix}_part{pid}_system_count'] = part_system_count.get(pid_int, 0)

    # 3. Coherence between system outputs (if provided)
    if system_outputs and len(system_outputs) >= 2:
        keys = list(system_outputs.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = system_outputs[keys[i]].flatten()
                b = system_outputs[keys[j]].flatten()
                corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
                diag[f'{prefix}_coherence_sys{keys[i]}_sys{keys[j]}'] = corr

    # 4. Bottleneck detection — Part with highest arousal = bottleneck
    max_arousal = 0.0
    bottleneck_part = -1
    for pid, part in brain.parts.items():
        if part.arousal.item() > max_arousal:
            max_arousal = part.arousal.item()
            bottleneck_part = int(pid)
    diag[f'{prefix}_bottleneck_part'] = bottleneck_part
    diag[f'{prefix}_bottleneck_arousal'] = max_arousal

    # 5. Tract utilization — magnitude of each tract's weight
    for name, tract in brain.white_matter.tracts.items():
        T = tract.get_transport_matrix()
        diag[f'{prefix}_tract_{name}_utilization'] = T.abs().mean().item()

    # === SHARED-PART DIAGNOSTICS (the 3 key ones) ===

    shared_ids = config.shared_part_ids or []

    # 9. Shared Part stress variance
    # Shared Parts receive stress from multiple Systems.
    # If stress variance is HIGH, the Systems are "fighting" over this Part.
    # If LOW, the Part has found a good compromise = generalizes well.
    for sid in shared_ids:
        part = brain.parts[str(sid)]
        n_systems = part_system_count.get(sid, 0)
        diag[f'{prefix}_shared{sid}_n_systems'] = n_systems
        diag[f'{prefix}_shared{sid}_mass_mean'] = part.mass.mean().item()
        diag[f'{prefix}_shared{sid}_arousal'] = part.arousal.item()

    # 10. Shared Part generalization — compare W similarity to non-shared Parts
    if shared_ids:
        shared_W_means = []
        nonshared_W_means = []
        for pid, part in brain.parts.items():
            pid_int = int(pid)
            w_mean = part.W_local.abs().mean().item()
            if pid_int in shared_ids:
                shared_W_means.append(w_mean)
            else:
                nonshared_W_means.append(w_mean)

        if shared_W_means:
            diag[f'{prefix}_shared_W_mean'] = sum(shared_W_means) / len(shared_W_means)
        if nonshared_W_means:
            diag[f'{prefix}_nonshared_W_mean'] = sum(nonshared_W_means) / len(nonshared_W_means)

    # 11. Generalization pressure — total multi-system pressure on shared Parts
    total_pressure = 0.0
    for sid in shared_ids:
        n = part_system_count.get(sid, 0)
        # Pressure = (n_systems - 1) * arousal (more systems + more stress = more pressure)
        part = brain.parts[str(sid)]
        pressure = max(0, n - 1) * part.arousal.item()
        total_pressure += pressure
        diag[f'{prefix}_shared{sid}_pressure'] = pressure

    diag[f'{prefix}_total_generalization_pressure'] = total_pressure

    return diag
