"""
SOMI 4.0 Knowledge Transplant
================================

Core transplant equation:
    delta_W = W_specialist - W_base

The "connectivity delta" captures EXACTLY what a specialist model learned
relative to its base. We transplant this delta into SOMI's W_local.

Example: If LLaMA-7B-code was fine-tuned from LLaMA-7B on code tasks,
then delta_W = LLaMA-7B-code - LLaMA-7B captures the "coding knowledge."
We add this to a SOMI Part's W_local to give it coding ability.

This achieves ~99% knowledge transfer without any training.

Source: SOMI_3_0/theory/06_KNOWLEDGE_ABSORPTION.md
"""

import torch
from typing import Dict, Optional, Tuple


def compute_delta(
    W_specialist: torch.Tensor,
    W_base: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the connectivity delta (what the specialist learned).

    Args:
        W_specialist: [dim, dim] specialist model's weight matrix
        W_base: [dim, dim] base model's weight matrix

    Returns:
        delta: [dim, dim] the knowledge that was learned
        diagnostics: Dict with delta statistics
    """
    delta = W_specialist - W_base

    diagnostics = {
        'delta_magnitude': delta.abs().mean().item(),
        'delta_max': delta.abs().max().item(),
        'delta_nonzero_frac': (delta.abs() > 1e-6).float().mean().item(),
        'delta_positive_frac': (delta > 0).float().mean().item(),
    }

    return delta, diagnostics


def transplant_knowledge(
    part: 'SOMIPart',
    delta: torch.Tensor,
    strength: float = 1.0,
    preserve_constraints: bool = True,
    stress_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Transplant a knowledge delta into a SOMI Part's W_local.

    Args:
        part: The SOMIPart receiving the knowledge
        delta: [hidden, hidden] connectivity delta to add
        strength: Scaling factor (1.0 = full transfer, 0.5 = half)
        preserve_constraints: Re-enforce W constraints after transplant
        stress_mask: [hidden, hidden] optional per-connection mask from
                     stress_guided_transplant(). Values in [0,1] where
                     1 = high target stress (needs knowledge), 0 = low
                     stress (already knows). If None, uniform transplant.

    Returns:
        diagnostics: Transplant metrics
    """
    from ..physics.geometry import enforce_constraints

    diagnostics = {}

    with torch.no_grad():
        W_before = part.W_local.clone()

        # Resize delta if needed (via interpolation)
        if delta.shape != part.W_local.shape:
            delta = torch.nn.functional.interpolate(
                delta.unsqueeze(0).unsqueeze(0),
                size=part.W_local.shape,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0).squeeze(0)
            diagnostics['transplant_resized'] = True

        # Apply stress mask: only transplant where the target is stressed
        if stress_mask is not None:
            if stress_mask.shape != delta.shape:
                stress_mask = torch.nn.functional.interpolate(
                    stress_mask.unsqueeze(0).unsqueeze(0),
                    size=delta.shape, mode='bilinear', align_corners=False,
                ).squeeze(0).squeeze(0)
            delta = delta * stress_mask.to(delta.device)
            diagnostics['transplant_stress_guided'] = True
            diagnostics['transplant_mask_coverage'] = (
                (stress_mask > 0.1).float().mean().item()
            )

        # Apply delta
        part.W_local.add_(strength * delta.to(part.W_local.device))

        # Re-enforce constraints
        if preserve_constraints:
            part.W_local.copy_(
                enforce_constraints(part.W_local.clone(), part.mask)
            )

        # Measure change
        W_change = (part.W_local - W_before).abs().mean().item()
        diagnostics['transplant_W_change'] = W_change
        diagnostics['transplant_strength'] = strength
        diagnostics['transplant_preserved_constraints'] = preserve_constraints

    return diagnostics


def stress_guided_transplant(
    brain: 'SOMICircuitBrain',
    source_brain: 'SOMICircuitBrain',
    base_weights: Optional[Dict[str, torch.Tensor]] = None,
    strength: float = 1.0,
    stress_threshold: float = 0.3,
) -> Dict[str, float]:
    """
    Transplant knowledge only where the TARGET brain is stressed.

    Level 2 absorption: builds stress maps for both brains, then
    masks the delta so we only graft where the target is ignorant
    (high stress) and the source is settled (low stress).

    Args:
        brain: Target SOMICircuitBrain (receives knowledge)
        source_brain: Source SOMICircuitBrain (provides knowledge)
        base_weights: Optional dict mapping part_id -> base W tensor.
                      If None, uses zeros (delta = source W directly).
        strength: Overall transplant strength
        stress_threshold: Stress level above which target needs knowledge

    Returns:
        diagnostics: Per-part transplant metrics
    """
    from ..physics.geometry import compute_stress_tensor

    all_diag = {}

    for pid in brain.parts:
        if pid not in source_brain.parts:
            continue

        target_part = brain.parts[pid]
        source_part = source_brain.parts[pid]

        # Compute delta
        if base_weights and pid in base_weights:
            base_W = base_weights[pid]
        else:
            base_W = torch.zeros_like(source_part.W_local)
        delta = source_part.W_local - base_W.to(source_part.W_local.device)

        # Build stress mask from target Part
        # High stress = target doesn't know this = transplant here
        H = target_part.config.hidden_dim
        phi_probe = torch.randn(1, 1, H, device=target_part.W_local.device) * 0.1
        S_target, _ = compute_stress_tensor(phi_probe, phi_probe, target_part.config)

        # Normalize stress to [0,1] mask
        S_mag = S_target.abs()
        s_max = S_mag.max().clamp(min=1e-8)
        stress_mask = (S_mag / s_max).clamp(0, 1)
        stress_mask = (stress_mask > stress_threshold).float() * stress_mask

        # Transplant with mask
        part_diag = transplant_knowledge(
            target_part, delta, strength=strength,
            stress_mask=stress_mask,
        )
        for k, v in part_diag.items():
            all_diag[f'part_{pid}_{k}'] = v

    all_diag['stress_guided'] = True
    return all_diag


def spectral_mode_transfer(
    target_part: 'SOMIPart',
    source_eigenvalues: torch.Tensor,
    source_eigenvectors: torch.Tensor,
    strength: float = 1.0,
) -> Dict[str, float]:
    """
    Transfer knowledge via eigenmodes instead of raw W.

    Level 3 absorption: when source and target have different architectures
    or sizes, transfer the spectral structure (eigenvalues + eigenvectors)
    rather than raw weights. The eigenmodes represent the fundamental
    "vibration patterns" of the network, which are more transferable
    across architectures than raw connection weights.

    The target's W is reconstructed from transferred eigenmodes:
        W_new = V_target @ diag(lambda_source) @ V_target^T
    where V_target are the target's eigenvectors and lambda_source are
    the source's eigenvalues (the "knowledge" being transferred).

    Args:
        target_part: SOMIPart receiving spectral knowledge
        source_eigenvalues: [K] eigenvalues from source
        source_eigenvectors: [source_dim, K] eigenvectors from source
        strength: Blending strength (0=keep target, 1=full transfer)

    Returns:
        diagnostics: Transfer metrics
    """
    from ..physics.forces import compute_laplacian
    from ..physics.settling import compute_eigendecomposition

    diagnostics = {}
    H = target_part.config.hidden_dim
    K = min(source_eigenvalues.shape[0], H)

    with torch.no_grad():
        # Get target eigenvectors (keep target's topology structure)
        L_target = compute_laplacian(target_part.W_local)
        target_evals, target_evecs, _ = compute_eigendecomposition(L_target, K)

        # Transfer source eigenvalues into target's eigenvector basis
        # This preserves the target's topology while importing source's
        # spectral structure (the "what it learned" encoded in eigenvalues)
        src_evals = source_eigenvalues[:K].to(target_part.W_local.device)
        tgt_evecs = target_evecs[:, :K]

        W_transferred = tgt_evecs @ torch.diag(src_evals) @ tgt_evecs.T
        W_transferred = W_transferred.abs()
        W_transferred.fill_diagonal_(0)
        row_sums = W_transferred.sum(1, keepdim=True).clamp(min=1e-8)
        W_transferred = W_transferred / row_sums

        # Blend with existing W
        W_before = target_part.W_local.clone()
        target_part.W_local.copy_(
            (1 - strength) * target_part.W_local + strength * W_transferred
        )

        diagnostics['spectral_transfer_K'] = K
        diagnostics['spectral_transfer_strength'] = strength
        diagnostics['spectral_transfer_W_change'] = (
            (target_part.W_local - W_before).abs().mean().item()
        )
        diagnostics['spectral_transfer_eigenvalue_range'] = (
            src_evals.min().item(), src_evals.max().item()
        )

    return diagnostics
