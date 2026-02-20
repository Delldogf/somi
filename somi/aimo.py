"""
SOMI AIMO 3 Pipeline — Consistency Scoring, Settle Check, Fusion
==================================================================

Uses SOMI physics during and after generation for better answer selection:

1. Consistency Scoring:
   Score how self-consistent a generation is by measuring stress across
   the hidden states. Low stress = self-consistent = trustworthy answer.

2. Post-Generation Settle Check:
   After generating an answer, run SOMI settling on the final state.
   If the state changes a lot (high phi_dot after settling), the answer
   was unstable — probably wrong. If it barely changes, the answer was
   already at equilibrium — probably right.

3. Confidence-Consistency Fusion:
   Combine the model's confidence (softmax probability) with SOMI's
   consistency score to pick the best answer from multiple candidates.

Theory: SOMI_3_0/theory/18_SOMI_TRAINING_COMPRESSION_AIMO3.md
"""

import torch
from typing import Dict, List, Optional, Tuple


def consistency_score(
    hidden_states: List[torch.Tensor],
) -> Tuple[float, Dict]:
    """
    Score self-consistency of a sequence of hidden states.

    Measures how much the hidden states change across the sequence.
    Large changes = inconsistent reasoning = low score.
    Small changes = consistent reasoning = high score.

    The "stress" here is the inter-step change magnitude, analogous
    to SOMI's stress tensor measuring prediction error.

    Args:
        hidden_states: List of [batch, hidden] tensors from each
                       generation step (or layer).

    Returns:
        score: Float in [0, 1] where 1 = perfectly consistent
        diagnostics: Detailed consistency metrics
    """
    if len(hidden_states) < 2:
        return 1.0, {'n_states': len(hidden_states)}

    changes = []
    for i in range(1, len(hidden_states)):
        delta = (hidden_states[i] - hidden_states[i-1]).abs().mean().item()
        changes.append(delta)

    avg_change = sum(changes) / len(changes)
    max_change = max(changes)

    score = 1.0 / (1.0 + avg_change)

    diagnostics = {
        'consistency_avg_change': avg_change,
        'consistency_max_change': max_change,
        'consistency_n_states': len(hidden_states),
        'consistency_score': score,
    }

    return score, diagnostics


def settle_check(
    brain: 'SOMICircuitBrain',
    final_hidden: torch.Tensor,
    n_settle_extra: int = 5,
) -> Tuple[float, Dict]:
    """
    Post-generation settle check: is the answer at equilibrium?

    After generating, pass the final hidden state through SOMI settling.
    If phi barely moves (low velocity), the answer is stable.
    If phi moves a lot (high velocity), the answer is unstable.

    Args:
        brain: SOMICircuitBrain
        final_hidden: [batch, seq, input_dim] final hidden state
        n_settle_extra: Additional settling steps to run

    Returns:
        stability: Float in [0, 1] where 1 = perfectly stable
        diagnostics: Settle check metrics
    """
    with torch.no_grad():
        h = brain.x_norm(brain.x_encoder(final_hidden))

        total_velocity = 0.0
        total_phi_change = 0.0

        for pid, part in brain.parts.items():
            phi_before = h.clone()
            phi, phi_dot, info = part(h, phi_target=h, training=False)

            velocity = phi_dot.abs().mean().item()
            phi_change = (phi - phi_before).abs().mean().item()

            total_velocity += velocity
            total_phi_change += phi_change

        n_parts = len(brain.parts)
        avg_velocity = total_velocity / max(n_parts, 1)
        avg_change = total_phi_change / max(n_parts, 1)

        stability = 1.0 / (1.0 + avg_velocity + avg_change)

    diagnostics = {
        'settle_check_avg_velocity': avg_velocity,
        'settle_check_avg_phi_change': avg_change,
        'settle_check_stability': stability,
        'settle_check_n_parts': n_parts,
    }

    return stability, diagnostics


def confidence_consistency_fusion(
    candidates: List[Dict],
    alpha: float = 0.5,
) -> Tuple[int, Dict]:
    """
    Pick the best answer by fusing confidence and consistency.

    For each candidate answer, combines:
      - Confidence: softmax probability (from the language model)
      - Consistency: SOMI consistency score (from physics)
      - Stability: settle check score (from physics)

    Final score = alpha * confidence + (1-alpha) * (consistency + stability) / 2

    Args:
        candidates: List of dicts, each containing:
            - 'confidence': float (softmax probability)
            - 'consistency': float (from consistency_score())
            - 'stability': float (from settle_check())
            - 'text': str (the answer text, optional)
        alpha: Weight between confidence and physics scores.
               0.5 = equal weight. Higher = trust softmax more.

    Returns:
        best_idx: Index of the best candidate
        diagnostics: Fusion metrics for all candidates
    """
    if not candidates:
        return 0, {}

    scores = []
    for c in candidates:
        conf = c.get('confidence', 0.5)
        cons = c.get('consistency', 0.5)
        stab = c.get('stability', 0.5)

        physics_score = (cons + stab) / 2.0
        fused = alpha * conf + (1 - alpha) * physics_score
        scores.append(fused)

    best_idx = scores.index(max(scores))

    diagnostics = {
        'fusion_scores': scores,
        'fusion_best_idx': best_idx,
        'fusion_best_score': scores[best_idx],
        'fusion_alpha': alpha,
        'fusion_n_candidates': len(candidates),
    }

    for i, (c, s) in enumerate(zip(candidates, scores)):
        diagnostics[f'candidate_{i}_confidence'] = c.get('confidence', 0)
        diagnostics[f'candidate_{i}_consistency'] = c.get('consistency', 0)
        diagnostics[f'candidate_{i}_stability'] = c.get('stability', 0)
        diagnostics[f'candidate_{i}_fused'] = s

    return best_idx, diagnostics
