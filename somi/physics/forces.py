"""
SOMI 4.0 Field Forces
=====================

Master field equation (what this file implements):

    M_i * phi_ddot_i + beta_i * phi_dot_i = F_total_i

Where F_total is the sum of 9 forces, each with a physical origin.

KEY CHANGES from somi_2_0/core.py:
  - Force 7 (coordination) is now LOCAL: 0.5 * lambda_C * phi
    NOT nonlocal: 0.5 * lambda_C * (phi @ W.T)
    This is the Level 2 fix from 24_THE_5_LEVELS_COMPLETE_REFERENCE.md
  - Every function returns (result, diagnostics_dict) — nothing happens silently
  - Uses SOMIBrainConfig instead of SOMIConfig

Reuses from somi_2_0/core.py:
  - compute_laplacian() — random-walk Laplacian L_rw = I - D^{-1}W
  - PrecisionTracker — precision weights from error variance
  - BasalGangliaGate — error gating bottleneck

Source equations: SOMI_3_0/theory/24_THE_5_LEVELS_COMPLETE_REFERENCE.md (Level 1)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .core import compute_laplacian, PrecisionTracker, BasalGangliaGate

# Re-export for convenience
__all__ = [
    'compute_laplacian',
    'PrecisionTracker',
    'BasalGangliaGate',
    'compute_field_forces',
    'compute_potential',
]


def compute_field_forces(
    phi: torch.Tensor,          # [batch, seq, hidden] or [hidden]
    phi_dot: torch.Tensor,      # same shape as phi
    phi_target: torch.Tensor,   # same shape as phi
    W: torch.Tensor,            # [hidden, hidden]
    L_rw: torch.Tensor,         # [hidden, hidden]
    precision: torch.Tensor,    # [hidden]
    beta: torch.Tensor,         # [hidden] or scalar — damping coefficient
    config: 'SOMIBrainConfig',
    gate: Optional[BasalGangliaGate] = None,
    training: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Compute ALL 9 forces acting on the field phi.

    Forces 1-7 come from the action (conservative: F = -dV/dphi).
    Forces 8-9 are non-conservative (damping, noise).

    KEY FIX: Force 7 (coordination) uses LOCAL formula:
        0.5 * lambda_C * phi  (correct, from Level 2 continuum limit)
    NOT the old nonlocal formula:
        0.5 * lambda_C * (phi @ W.T)  (wrong, had 8% extra coupling)

    Args:
        phi: Activity field
        phi_dot: Velocity field
        phi_target: Prediction target (from JEPA Y-encoder or previous state)
        W: Connectivity matrix [hidden, hidden]
        L_rw: Random-walk Laplacian (from compute_laplacian)
        precision: Precision weights per feature [hidden] (from PrecisionTracker)
        beta: Damping coefficient per feature [hidden] or scalar
        config: SOMIBrainConfig
        gate: Optional BasalGangliaGate for error filtering
        training: Whether in training mode (enables noise)

    Returns:
        forces: Dict of {name: tensor} for each force + 'total'
        diagnostics: Dict of {name: float} with per-force magnitudes
    """
    error = phi - phi_target  # Prediction error

    forces = {}

    # ---- Forces from the action (F = -dV/dphi) ----

    # 1. COUPLING: Activity spreads across connected features
    #    From V_coupling = (alpha_1 / 2) * phi^T @ L_rw @ phi
    #    Brain analog: synaptic coupling smooths representations
    forces['coupling'] = -config.alpha_1 * (phi @ L_rw.T)

    # 2. ANCHORING: Restoring force toward zero
    #    From V_anchor = (alpha_0 / 2) * |phi|^2
    #    Brain analog: leak conductance prevents drift
    #    Level 3 interpretation: alpha_0 = cosmological constant (dark energy)
    forces['anchor'] = -config.alpha_0 * phi

    # 3. NONLINEAR SATURATION: Prevents unbounded growth
    #    From V_nonlinear = sum(ln(cosh(phi_i)))
    #    Brain analog: firing rate saturation
    #    NOTE: tanh(x) ≈ x - x^3/3 creates cross-frequency coupling (CFC)
    #    automatically. Do NOT add a separate CFC force — it double-counts.
    forces['nonlinear'] = -torch.tanh(phi.clamp(-10, 10))

    # 4. INFO STRESS: Precision-weighted prediction error
    #    From V_info = (kappa_0 / 2) * sum(Pi_i * (phi_i - phi_hat_i)^2)
    #    Brain analog: cortical columns correct predictions weighted by precision
    # Broadcast precision to match phi shape
    if phi.dim() == 3:
        prec = precision.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
    elif phi.dim() == 2:
        prec = precision.unsqueeze(0)  # [1, hidden]
    else:
        prec = precision  # [hidden]
    forces['info'] = -config.kappa_0 * prec * error

    # 5. BASAL GANGLIA GATE: Filters small/noisy errors
    #    Not from the action (external drive)
    #    Brain analog: basal ganglia selects which errors matter
    #    Uses fixed random projections (not learned) — 248x selectivity
    if gate is not None and config.kappa_1 > 0:
        forces['gate'] = -config.kappa_1 * gate(error)
    else:
        forces['gate'] = torch.zeros_like(phi)

    # 6. ERROR SMOOTHING: Errors diffuse to neighbors
    #    From V_error_smooth = (lambda_E / 2) * e^T @ L_rw @ e
    #    Brain analog: lateral inhibition shares error responsibility
    forces['error_smooth'] = -config.lambda_E * (error @ L_rw.T)

    # 7. COORDINATION: Features co-activate (LOCAL — Level 2 fix!)
    #    OLD (wrong): 0.5 * lambda_C * (phi @ W.T)  <-- nonlocal, 8% extra coupling
    #    NEW (correct): 0.5 * lambda_C * phi         <-- local, from continuum limit
    #
    #    Level 2 proof: In the continuum limit, the coordination potential
    #    V_coord = -(lambda_C/4) * phi^2 is LOCAL (no spatial coupling).
    #    The old (phi @ W.T) formula added nonlocal coupling that shouldn't be there.
    #    See: 24_THE_5_LEVELS_COMPLETE_REFERENCE.md, Level 2 -> Level 1 upgrades
    forces['coordination'] = 0.5 * config.lambda_C * phi

    # ---- Forces NOT from the action (dissipative/stochastic) ----

    # 8. DAMPING: Energy dissipation
    #    From Rayleigh dissipation R = (1/2) * phi_dot^T @ B(W) @ phi_dot
    #    Brain analog: membrane resistance drains kinetic energy
    #    beta can be scalar or per-feature tensor
    if isinstance(beta, (int, float)):
        forces['damping'] = -beta * phi_dot
    elif beta.dim() == 0:
        forces['damping'] = -beta * phi_dot
    else:
        # Per-feature damping: beta [hidden], phi_dot [batch, seq, hidden] or [hidden]
        if phi_dot.dim() > beta.dim():
            for _ in range(phi_dot.dim() - beta.dim()):
                beta = beta.unsqueeze(0)
        forces['damping'] = -beta * phi_dot

    # 9. NOISE: Exploration
    #    Stochastic drive (Langevin dynamics)
    #    Brain analog: spontaneous neural activity for exploration
    if training and config.noise_ratio > 0:
        scale = config.noise_ratio * phi.detach().std().clamp(min=1e-6)
        forces['noise'] = scale * torch.randn_like(phi)
    else:
        forces['noise'] = torch.zeros_like(phi)

    # Total force (sum of all 9)
    forces['total'] = sum(f for k, f in forces.items() if k != 'total')

    # ---- Diagnostics (nothing happens silently) ----
    diagnostics = {}
    for name, f in forces.items():
        mag = f.detach().abs().mean().item()
        diagnostics[f'force_{name}_magnitude'] = mag
    # Also track error magnitude for monitoring
    diagnostics['error_magnitude'] = error.detach().abs().mean().item()
    diagnostics['phi_magnitude'] = phi.detach().abs().mean().item()
    diagnostics['phi_dot_magnitude'] = phi_dot.detach().abs().mean().item()

    return forces, diagnostics


def compute_potential(
    phi: torch.Tensor,
    phi_target: torch.Tensor,
    W: torch.Tensor,
    L_rw: torch.Tensor,
    precision: torch.Tensor,
    config: 'SOMIBrainConfig',
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute total potential V(phi, W) from the SOMI action.

    V = V_coupling + V_anchor + V_nonlinear + V_info + V_error_smooth + V_coord

    Each term corresponds to one line in the action functional.

    Returns:
        V: scalar (total potential, averaged over batch and sequence)
        diagnostics: Dict with per-term contributions
    """
    error = phi - phi_target

    # Broadcast precision
    if phi.dim() == 3:
        prec = precision.unsqueeze(0).unsqueeze(0)
    elif phi.dim() == 2:
        prec = precision.unsqueeze(0)
    else:
        prec = precision

    terms = {}

    # V_coupling = (alpha_1/2) * phi^T @ L_rw @ phi
    terms['coupling'] = 0.5 * config.alpha_1 * (phi * (phi @ L_rw.T)).sum(dim=-1).mean()

    # V_anchor = (alpha_0/2) * |phi|^2
    terms['anchor'] = 0.5 * config.alpha_0 * (phi ** 2).sum(dim=-1).mean()

    # V_nonlinear = sum(ln(cosh(phi_i)))
    terms['nonlinear'] = torch.log(torch.cosh(phi.clamp(-10, 10))).sum(dim=-1).mean()

    # V_info = (kappa_0/2) * sum(Pi_i * (phi_i - phi_hat_i)^2)
    terms['info'] = 0.5 * config.kappa_0 * (prec * error ** 2).sum(dim=-1).mean()

    # V_error_smooth = (lambda_E/2) * e^T @ L_rw @ e
    terms['error_smooth'] = 0.5 * config.lambda_E * (error * (error @ L_rw.T)).sum(dim=-1).mean()

    # V_coord = -(lambda_C/4) * |phi|^2  (LOCAL — Level 2 fix)
    # OLD: -(lambda_C/4) * phi^T @ W @ phi  (nonlocal — wrong)
    terms['coord'] = -0.25 * config.lambda_C * (phi ** 2).sum(dim=-1).mean()

    # Total potential
    V = sum(terms.values())

    # Diagnostics
    diagnostics = {f'V_{name}': val.item() for name, val in terms.items()}
    diagnostics['V_total'] = V.item()

    return V, diagnostics
