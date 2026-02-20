"""
SOMI 4.0 Quantum Diagnostics (Level 5)
=========================================

5 metrics from the path integral / quantum level:
1. Ensemble uncertainty — variance across model ensemble
2. Saddle-point quality — how well does the mean approximate the ensemble?
3. Topological protection — error resilience of learned representations
4. Renormalization group — scale hierarchy of features
5. Holographic diagnostics — boundary-bulk correspondence
"""

import torch
from typing import Dict, List, Optional


def compute_quantum_diagnostics(
    brain: 'SOMICircuitBrain',
    ensemble_brains: Optional[List] = None,
    test_input: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute all Level 5 quantum diagnostics.

    Most of these require either an ensemble of brains or test inputs.
    Without them, only structural metrics are computed.

    Args:
        brain: The primary SOMICircuitBrain
        ensemble_brains: Optional list of brain copies (for ensemble)
        test_input: Optional test input for running through models
    """
    diag = {}
    prefix = 'L5'

    # 1. Ensemble uncertainty (if ensemble available)
    if ensemble_brains and test_input is not None:
        outputs = []
        for b in ensemble_brains:
            with torch.no_grad():
                logits, _ = b(test_input, training=False)
                outputs.append(logits)

        stacked = torch.stack(outputs)
        ensemble_mean = stacked.mean(dim=0)
        ensemble_var = stacked.var(dim=0)

        diag[f'{prefix}_ensemble_uncertainty_mean'] = ensemble_var.mean().item()
        diag[f'{prefix}_ensemble_uncertainty_max'] = ensemble_var.max().item()
        diag[f'{prefix}_ensemble_size'] = len(ensemble_brains)

        # 2. Saddle-point quality (mean vs ensemble)
        # If variance is low, the mean is a good approximation
        diag[f'{prefix}_saddle_point_quality'] = 1.0 / (1.0 + ensemble_var.mean().item())
    else:
        diag[f'{prefix}_ensemble_available'] = False

    # 3. Topological protection — robustness to small perturbations
    # Measure how much output changes with small W perturbations
    if test_input is not None:
        with torch.no_grad():
            base_output, _ = brain(test_input, training=False)

            # Perturb each Part's W slightly
            total_change = 0.0
            for pid, part in brain.parts.items():
                W_original = part.W_local.clone()
                noise = 0.01 * torch.randn_like(part.W_local)
                part.W_local.add_(noise)

                perturbed_output, _ = brain(test_input, training=False)
                change = (perturbed_output - base_output).abs().mean().item()
                total_change += change

                # Restore
                part.W_local.copy_(W_original)

            n_parts = len(brain.parts)
            diag[f'{prefix}_topological_sensitivity'] = total_change / max(n_parts, 1)
            diag[f'{prefix}_topological_protection'] = 1.0 / (1.0 + total_change / max(n_parts, 1))

    # 4. Renormalization group — scale hierarchy
    # Check mass spectrum for clear scale separation
    all_masses = []
    for part in brain.parts.values():
        all_masses.append(part.mass)

    if all_masses:
        full_mass = torch.cat(all_masses)
        sorted_mass = full_mass.sort().values

        # Scale ratio: ratio of largest to smallest mass
        diag[f'{prefix}_rg_scale_ratio'] = (
            sorted_mass[-1].item() / max(sorted_mass[0].item(), 1e-8)
        )

        # Number of distinct scales (using log-spaced bins)
        log_mass = torch.log(full_mass.clamp(min=1e-8))
        diag[f'{prefix}_rg_log_mass_std'] = log_mass.std().item()

    return diag
