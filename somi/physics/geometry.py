"""
SOMI 4.0 Geometry Equation
===========================

Master geometry equation (what this file implements):

    W_dot = -eta * (S + K_kinetic) - lambda_W * W

Where S is the 4-component information stress tensor:
    S_ij = (alpha_1/2)<(phi_i - phi_j)^2>          (activity mismatch)
         + (lambda_E/2)<(e_i - e_j)^2>             (error mismatch)
         - (lambda_C/4)<phi_i * phi_j>              (Hebbian correlation)
         - (lambda_C*kappa_stdp/4)<phi_i * phi_dot_j>  (STDP — ASYMMETRIC)

And K_kinetic is the kinetic stress from geometry-dependent mass:
    K_ij = -(M_i / h_i) * W_ij * <phi_dot_i^2>

KEY ENHANCEMENTS over somi_2_0/geometry.py:
  - Titans momentum on stress (smooths noisy geometry updates)
  - Mass-conductivity constraint: rho * kappa = 1/alpha_1
  - Data-dependent forgetting (alpha_t ~ 0 for same context, ~1 for change)
  - Stress-driven annealing (new learning from physics Levels 2-3)
  - Dale's Law via Signed Sinkhorn (not just non-negative clamp)
  - Returns diagnostics from every function

Reuses from somi_2_0/geometry.py:
  - compute_stress_tensor() — 4-component stress computation
  - compute_kinetic_stress() — mass-dependent kinetic stress
  - enforce_constraints() — W constraint enforcement
  - initialize_W() — sparse directed W initialization
  - structural_plasticity() — prune/grow connections

Source: SOMI_3_0/theory/24_THE_5_LEVELS_COMPLETE_REFERENCE.md (Levels 1-2)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .geometry_base import (
    compute_stress_tensor as _base_stress_tensor,
    compute_kinetic_stress,
    enforce_constraints,
    initialize_W,
    structural_plasticity,
    sinkhorn_normalize,
)

# Re-export for convenience
__all__ = [
    'compute_stress_tensor',
    'compute_kinetic_stress',
    'enforce_constraints',
    'initialize_W',
    'structural_plasticity',
    'signed_sinkhorn',
    'geometry_step',
    'StressMomentum',
]


def compute_stress_tensor(
    phi: torch.Tensor,
    phi_target: torch.Tensor,
    config: 'SOMIBrainConfig',
    phi_dot: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the 4-component information stress tensor S_ij.

    This wraps somi_2_0's compute_stress_tensor but:
    1. Works with SOMIBrainConfig (not SOMIConfig)
    2. Returns diagnostics dict alongside the tensor

    The stress tensor tells each connection how to change:
    - High stress = "this connection is wrong" = weaken it
    - Low/negative stress = "this connection is good" = strengthen it

    Think of it like tension in a rubber band: too much tension (stress)
    and the band snaps (connection weakens). The right amount of tension
    and it holds things together (connection strengthens).

    Args:
        phi: [batch, seq, hidden] settled field activity
        phi_target: [batch, seq, hidden] target (from JEPA or previous state)
        config: SOMIBrainConfig
        phi_dot: [batch, seq, hidden] velocity (for STDP). Optional.

    Returns:
        S: [hidden, hidden] stress tensor
        diagnostics: Dict with stress statistics
    """
    # The base function expects a SOMIConfig-like object, but our config
    # has the same field names (alpha_1, lambda_E, lambda_C, kappa_stdp)
    S = _base_stress_tensor(phi, phi_target, config, phi_dot)

    # Diagnostics
    diagnostics = {
        'stress_mean': S.mean().item(),
        'stress_std': S.std().item(),
        'stress_max': S.max().item(),
        'stress_min': S.min().item(),
        'stress_positive_frac': (S > 0).float().mean().item(),
        'stress_frobenius': S.norm().item(),
    }

    return S, diagnostics


class StressMomentum:
    """
    Momentum buffer for stress tensor (Titans paper enhancement).

    Instead of using raw stress S each step (which is noisy because it's
    computed from a single batch), we keep a running average:

        S_smoothed = beta * S_smoothed_prev + (1 - beta) * S_current

    This is like exponential moving average (EMA) for the stress.

    Why it helps:
    - Noisy stress -> geometry jumps around -> bad learning
    - Smoothed stress -> geometry changes gradually -> stable learning
    - Same idea as momentum in gradient descent (Adam, SGD with momentum)

    From Titans paper (Behrouz 2024): momentum on long-term memory updates
    reduced oscillation and improved convergence.

    Args:
        hidden_dim: Size of the stress tensor (hidden x hidden)
        beta: Momentum coefficient (0.9 = strong smoothing, 0.0 = no smoothing)
        device: Torch device
    """

    def __init__(self, hidden_dim: int, beta: float = 0.9, device: torch.device = None):
        self.beta = beta
        self.device = device or torch.device('cpu')
        self.S_ema = torch.zeros(hidden_dim, hidden_dim, device=self.device)
        self.initialized = False

    def update(self, S_current: torch.Tensor) -> torch.Tensor:
        """
        Apply momentum and return smoothed stress.

        Args:
            S_current: [hidden, hidden] current raw stress tensor

        Returns:
            S_smoothed: [hidden, hidden] EMA-smoothed stress tensor
        """
        with torch.no_grad():
            if not self.initialized:
                self.S_ema = S_current.clone()
                self.initialized = True
            else:
                self.S_ema = self.beta * self.S_ema + (1 - self.beta) * S_current
        return self.S_ema.clone()


def signed_sinkhorn(
    W: torch.Tensor,
    ei_mask: torch.Tensor,
    n_iters: int = 10,
) -> torch.Tensor:
    """
    Signed Sinkhorn normalization for Dale's Law (E/I balance).

    Dale's Law: Each neuron is either excitatory OR inhibitory, never both.
    In the brain, ~80% of neurons are excitatory (glutamate) and ~20% are
    inhibitory (GABA).

    Instead of just clamping W >= 0 (which loses inhibitory connections),
    we split W into excitatory and inhibitory components:
        W_effective = W_excitatory - W_inhibitory

    Both W_excitatory and W_inhibitory are non-negative.
    The ei_mask determines which features are excitatory (+1) vs inhibitory (-1).

    Signed Sinkhorn normalizes both components so:
    - Total excitation and inhibition are balanced
    - No single feature dominates

    Args:
        W: [hidden, hidden] raw connectivity (can have any sign)
        ei_mask: [hidden] with +1 for excitatory features, -1 for inhibitory
        n_iters: Sinkhorn normalization iterations

    Returns:
        W_signed: [hidden, hidden] Dale's Law compliant W
    """
    with torch.no_grad():
        # Separate excitatory and inhibitory components
        exc_cols = (ei_mask > 0)  # Which features are excitatory
        inh_cols = (ei_mask < 0)  # Which features are inhibitory

        # Excitatory submatrix: connections FROM excitatory features
        W_exc = W[:, exc_cols].clamp(min=0)
        # Inhibitory submatrix: connections FROM inhibitory features (flip sign)
        W_inh = (-W[:, inh_cols]).clamp(min=0)

        # Sinkhorn normalize each component
        if W_exc.numel() > 0:
            row_sums = W_exc.sum(dim=1, keepdim=True).clamp(min=1e-8)
            W_exc = W_exc / row_sums  # Row normalize
        if W_inh.numel() > 0:
            row_sums = W_inh.sum(dim=1, keepdim=True).clamp(min=1e-8)
            W_inh = W_inh / row_sums

        # Reconstruct: W_effective = W_exc - W_inh
        W_new = W.clone()
        W_new[:, exc_cols] = W_exc
        W_new[:, inh_cols] = -W_inh  # Inhibitory connections are negative

        W_new.fill_diagonal_(0)

    return W_new


def mass_conductivity_constraint(
    mass: torch.Tensor,
    W: torch.Tensor,
    alpha_1: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Enforce mass-conductivity duality: rho * kappa = 1 / alpha_1.

    Level 2 discovery: In the continuum limit, mass density (rho) and
    conductivity (kappa) are inversely related through:

        rho_i * kappa_i = 1 / alpha_1

    Where:
        rho_i = mass of feature i
        kappa_i = total connection strength of feature i (row sum of W)

    What this means:
    - Heavy features (high mass, slow to change) must have low connectivity
    - Light features (low mass, fast to change) must have high connectivity
    - You can't have both heavy AND well-connected — that would violate
      the continuum physics

    Brain analog: Highly connected hub neurons (high kappa) tend to be fast
    responders (low mass). Specialized deep neurons (high mass) have fewer
    but stronger connections.

    Args:
        mass: [hidden] per-feature mass
        W: [hidden, hidden] connectivity
        alpha_1: Coupling strength

    Returns:
        mass_adjusted: [hidden] mass satisfying the constraint
        diagnostics: Constraint satisfaction metrics
    """
    with torch.no_grad():
        kappa = W.abs().sum(dim=1).clamp(min=1e-8)  # [hidden]
        target_mass = 1.0 / (alpha_1 * kappa)  # rho * kappa = 1/alpha_1

        # Soft constraint: blend toward target
        mass_adjusted = 0.9 * mass + 0.1 * target_mass

        # Diagnostics
        product = mass_adjusted * kappa
        expected = 1.0 / alpha_1
        violation = (product - expected).abs().mean().item()

        diagnostics = {
            'mass_conductivity_violation': violation,
            'mass_mean': mass_adjusted.mean().item(),
            'mass_std': mass_adjusted.std().item(),
            'kappa_mean': kappa.mean().item(),
            'kappa_std': kappa.std().item(),
        }

    return mass_adjusted, diagnostics


def compute_data_dependent_alpha(
    S_current: torch.Tensor,
    S_previous: torch.Tensor,
) -> float:
    """
    Data-dependent forgetting factor from Titans paper.

    The idea: When the current data is similar to previous data (low stress
    change), keep learning normally (alpha ~ 0, don't forget).
    When the data changes dramatically (high stress change), forget old
    patterns and learn fresh (alpha ~ 1, full forgetting).

    This is like how your brain forgets the layout of a hotel room
    when you check into a new one, but remembers your home layout perfectly
    because you see it every day.

    Formula:
        alpha_t = sigmoid(||S_current - S_previous|| / ||S_previous|| - 1)

    Returns:
        alpha: float in [0, 1]. 0 = retain everything, 1 = forget everything.
    """
    with torch.no_grad():
        if S_previous is None or S_previous.norm() < 1e-8:
            return 0.0

        relative_change = (S_current - S_previous).norm() / (S_previous.norm() + 1e-8)
        alpha = torch.sigmoid(relative_change - 1.0).item()

    return alpha


def geometry_step(
    W: torch.Tensor,
    S: torch.Tensor,
    eta: float,
    config: 'SOMIBrainConfig',
    mask: Optional[torch.Tensor] = None,
    K_kinetic: Optional[torch.Tensor] = None,
    ei_mask: Optional[torch.Tensor] = None,
    forget_alpha: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    One geometry update step (enhanced for SOMI 4.0).

    Update rule:
        dW = -eta * (1 - forget_alpha) * (S + K_kinetic) - lambda_W * W

    When forget_alpha > 0 (data context changed), the stress signal is
    reduced — the model "forgets" old patterns and is more open to new ones.

    After the update, enforces constraints:
    1. Dale's Law via Signed Sinkhorn (if ei_mask provided)
    2. Standard constraints (non-negative, zero diagonal, row/col caps)
    3. Sparsity mask

    Args:
        W: [hidden, hidden] current connectivity
        S: [hidden, hidden] smoothed stress tensor (after momentum)
        eta: Learning rate (from calibration)
        config: SOMIBrainConfig
        mask: [hidden, hidden] sparsity mask
        K_kinetic: [hidden, hidden] kinetic stress (optional)
        ei_mask: [hidden] E/I sign mask (optional, for Dale's Law)
        forget_alpha: Data-dependent forgetting [0,1]

    Returns:
        W_new: [hidden, hidden] updated connectivity
        diagnostics: Dict with update statistics
    """
    with torch.no_grad():
        # Total stress = information stress + kinetic stress
        total_stress = S.clone()
        if K_kinetic is not None:
            total_stress = total_stress + K_kinetic

        # Data-dependent forgetting: reduce stress signal in new contexts
        effective_stress = (1.0 - forget_alpha) * total_stress

        # Compute update
        dW = -eta * effective_stress - config.lambda_W * W

        # Apply sparsity mask
        if mask is not None:
            dW = dW * mask

        # Apply update
        W_new = W + dW

        # Enforce constraints
        if config.dales_law and ei_mask is not None:
            W_new = signed_sinkhorn(W_new, ei_mask)
        else:
            W_new = enforce_constraints(W_new, mask)

        # Diagnostics
        dW_magnitude = dW.abs().mean().item()
        W_magnitude = W_new.abs().mean().item()
        W_sparsity = (W_new.abs() < 1e-6).float().mean().item()

        diagnostics = {
            'geometry_dW_magnitude': dW_magnitude,
            'geometry_W_magnitude': W_magnitude,
            'geometry_W_sparsity': W_sparsity,
            'geometry_eta': eta,
            'geometry_forget_alpha': forget_alpha,
            'geometry_stress_norm': total_stress.norm().item(),
        }
        if K_kinetic is not None:
            diagnostics['geometry_kinetic_norm'] = K_kinetic.norm().item()

    return W_new, diagnostics
