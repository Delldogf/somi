"""
SOMI 4.0 Knowledge Fingerprinting — Full Physics Fingerprint
===============================================================

A fingerprint captures "what does this model know?" using three signals:
  1. Probe outputs (confidence, entropy on test inputs)
  2. Stress profile (per-Part stress map — where is it struggling?)
  3. Eigenspectrum (spectral structure of each Part's W)
  4. Wilson loops (topological invariants — gauge-invariant knowledge)

knowledge_diff() compares two fingerprints to answer:
  "What does Model B know that Model A doesn't?"
"""

import torch
from typing import Dict, List, Optional, Tuple

from ..physics.forces import compute_laplacian
from ..physics.settling import compute_eigendecomposition
from ..physics.geometry import compute_stress_tensor


def compute_fingerprint(
    brain: 'SOMICircuitBrain',
    probe_inputs: Optional[List[torch.Tensor]] = None,
    probe_labels: Optional[List[torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Compute a full physics fingerprint of the brain.

    Includes:
      - Probe confidence/entropy (if probe_inputs given)
      - Per-Part stress profile (mean, max, std of stress tensor)
      - Per-Part eigenspectrum (spectral gap, energy distribution)
      - Wilson loop values (topological invariants)

    Args:
        brain: The SOMICircuitBrain to fingerprint
        probe_inputs: Optional list of [batch, seq, input_dim] probes
        probe_labels: Optional list of expected output tensors

    Returns:
        fingerprint: Dict with all fingerprint signals
    """
    fp = {}

    # === 1. Probe outputs (if provided) ===
    if probe_inputs is not None:
        with torch.no_grad():
            for i, probe in enumerate(probe_inputs):
                logits, _ = brain(probe, training=False)

                if probe_labels is not None and i < len(probe_labels):
                    probs = torch.softmax(logits, dim=-1)
                    target = probe_labels[i]
                    if target.dim() < probs.dim():
                        target = target.unsqueeze(-1)
                    correct_probs = probs.gather(
                        -1, target.clamp(0, probs.shape[-1] - 1)
                    )
                    fp[f'probe_{i}_confidence'] = correct_probs.mean().item()
                else:
                    fp[f'probe_{i}_output_magnitude'] = (
                        logits.abs().mean().item()
                    )

                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum(-1).mean().item()
                fp[f'probe_{i}_entropy'] = entropy

        fp['n_probes'] = len(probe_inputs)

    # === 2. Stress profile per Part ===
    for pid, part in brain.parts.items():
        H = part.config.hidden_dim
        phi_probe = torch.randn(1, 1, H, device=part.W_local.device) * 0.1
        S, _ = compute_stress_tensor(phi_probe, phi_probe, part.config)
        S_mag = S.abs()
        fp[f'part_{pid}_stress_mean'] = S_mag.mean().item()
        fp[f'part_{pid}_stress_max'] = S_mag.max().item()
        fp[f'part_{pid}_stress_std'] = S_mag.std().item()

    # === 3. Eigenspectrum per Part ===
    for pid, part in brain.parts.items():
        L_rw = compute_laplacian(part.W_local)
        evals, _, ediag = compute_eigendecomposition(L_rw)
        fp[f'part_{pid}_spectral_gap'] = ediag.get('eigen_spectral_gap', 0)
        fp[f'part_{pid}_eigen_max'] = ediag.get('eigen_max', 0)
        fp[f'part_{pid}_eigen_median'] = ediag.get('eigen_median', 0)
        pos = evals[evals > 1e-8]
        if len(pos) > 0:
            total_e = pos.sum().item()
            cumul = torch.cumsum(pos, 0) / total_e
            rank_95 = (cumul >= 0.95).nonzero()
            fp[f'part_{pid}_spectral_rank_95'] = (
                rank_95[0].item() + 1 if len(rank_95) > 0 else len(pos)
            )
        else:
            fp[f'part_{pid}_spectral_rank_95'] = 0

    # === 4. Wilson loops (topological invariants) ===
    for route in (brain.config.system_routes or []):
        if len(route) >= 3:
            loop = route + [route[0]]
            try:
                _, wilson_diag = brain.white_matter.compute_wilson_loop(loop)
                for k, v in wilson_diag.items():
                    fp[f'wilson_{k}'] = v
            except Exception:
                pass

    # === 5. Global metrics ===
    fp['n_parts'] = brain.config.n_parts
    fp['hidden_dim'] = brain.config.hidden_dim
    fp['n_systems'] = len(brain.systems)

    return fp


def compare_fingerprints(
    before: Dict[str, float],
    after: Dict[str, float],
) -> Dict[str, float]:
    """
    Compare fingerprints before and after absorption.

    Returns:
        comparison: Dict with gain/loss for each probe
    """
    comparison = {}
    for key in before:
        if key in after and isinstance(before[key], (int, float)):
            change = after[key] - before[key]
            comparison[f'{key}_change'] = change
            comparison[f'{key}_improved'] = change > 0
    return comparison


def knowledge_diff(
    fp_a: Dict[str, float],
    fp_b: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute "what does B know that A doesn't?"

    Compares two fingerprints to find:
      - Where B has lower stress than A (B is more confident)
      - Where B has richer eigenspectrum (more structured knowledge)
      - Where B has better probe scores (higher accuracy)

    Args:
        fp_a: Fingerprint of model A (the receiver)
        fp_b: Fingerprint of model B (the teacher)

    Returns:
        diff: Dict describing knowledge that B has but A lacks.
              Positive values = B is better at this.
    """
    diff = {}

    # Stress differences: lower stress = more knowledge
    for key in fp_b:
        if 'stress_mean' in key and key in fp_a:
            diff[f'{key}_gap'] = fp_a[key] - fp_b[key]

    # Spectral differences: higher spectral gap = more structured
    for key in fp_b:
        if 'spectral_gap' in key and key in fp_a:
            diff[f'{key}_advantage'] = fp_b[key] - fp_a[key]

    # Probe differences: higher confidence = more knowledge
    for key in fp_b:
        if 'confidence' in key and key in fp_a:
            diff[f'{key}_advantage'] = fp_b[key] - fp_a[key]

    # Entropy differences: lower entropy = more certain
    for key in fp_b:
        if 'entropy' in key and key in fp_a:
            diff[f'{key}_gap'] = fp_a[key] - fp_b[key]

    # Summary: which Parts should absorb from B?
    parts_needing_knowledge = []
    for key, val in diff.items():
        if 'stress_mean' in key and '_gap' in key and val > 0:
            part_id = key.split('_')[1]
            parts_needing_knowledge.append(part_id)
    diff['parts_needing_knowledge'] = parts_needing_knowledge
    diff['n_parts_needing_knowledge'] = len(parts_needing_knowledge)

    return diff
