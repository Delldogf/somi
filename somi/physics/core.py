"""
SOMI 2.0 Core Physics
=====================

The potential V(phi, W), forces F = -dV/dphi, and Hamiltonian H = T + V.

Everything traces back to the SOMI 2.0 unified action (Complete Theory):

    S[phi, W] = integral_0^T [ (1/2) phi_dot^T G(W) phi_dot - V(phi, W) ] dt

where G(W) is the geometry-dependent metric tensor. In the diagonal
approximation (implemented here):

    G(W) = diag(M_1(W), ..., M_n(W))

with M_i(W) = M_0 * h_mean / h_i(W) and h_i = sum_j W_ij^2 (Herfindahl index).

The potential V contains 7 terms, each with a brain analog.
Forces are -dV/dphi (from the action) plus damping (from Rayleigh dissipation).
The Hamiltonian H = T + V must decrease monotonically (Lyapunov stability).

Key equation references:
  - Field equation: M_i(W) phi_i_ddot + beta_i(W) phi_i_dot = -dV/dphi_i
  - Geometry equation: W_dot = -eta (S + K^kinetic) - lambda_W W
  - Stability: dH/dt = -phi_dot^T B(W) phi_dot <= 0

Source: SOMI_2_0_COMPLETE_THEORY.md
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from ..config import SOMIBrainConfig as SOMIConfig


# ============================================================
# Laplacian
# ============================================================

def compute_laplacian(W: torch.Tensor) -> torch.Tensor:
    """
    Compute random-walk Laplacian: L_rw = I - D_out^{-1} W

    D_out is the diagonal out-degree matrix (row sums of W). This
    normalization ensures P = D_out^{-1} W is row-stochastic (rows sum
    to 1), so eigenvalues of L_rw = I - P lie in the disk |lambda-1| <= 1.

    With ASYMMETRIC W (Feb 2026): L_rw is no longer symmetric, so it can
    have complex eigenvalues. However, for non-negative W with positive
    out-degrees, all eigenvalues still have non-negative real part.
    The imaginary parts create circulatory (wave-like) dynamics rather
    than pure dissipation.

    Stability: ChatGPT verified (Feb 2026) that with sufficient damping
    (beta^2 > 2*alpha*m), the system is asymptotically stable even with
    complex Laplacian eigenvalues.

    Args:
        W: [hidden_dim, hidden_dim] connectivity matrix (may be asymmetric)

    Returns:
        L_rw: [hidden_dim, hidden_dim] random-walk Laplacian
    """
    n = W.shape[0]
    # For safety, handle non-exactly-doubly-stochastic W
    row_sums = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
    D_inv_W = W / row_sums  # D⁻¹W
    L_rw = torch.eye(n, device=W.device, dtype=W.dtype) - D_inv_W
    return L_rw


# ============================================================
# Components: Precision Tracker and Basal Ganglia Gate
# ============================================================

class PrecisionTracker(nn.Module):
    """
    Tracks prediction error variance per feature and outputs precision weights.

    What it does:
      - Watches how much each feature's prediction error bounces around
      - Features with consistent errors (low variance) get HIGH precision
      - Features with noisy errors (high variance) get LOW precision
      - High precision means "pay attention to errors here — they're meaningful"
      - Low precision means "ignore errors here — they're just noise"

    Self-normalizing (homeostatic plasticity, Lesson #6):
      Π_i = (1/σ²_i) / mean(1/σ²_j)
      Average precision always = 1.0. Individual precisions vary.
      This prevents precision from changing the total signal magnitude.

    Second-order confidence (Doc 28):
      Also tracks how STABLE the precision estimates are. If precision
      estimates are volatile (changing a lot), trust them less. This is
      like the brain tracking "how confident am I in my confidence?"

    Brain analog: Cortical columns estimate reliability of their predictions
    from recent error history. More reliable columns get amplified signals.
    """

    def __init__(self, hidden_dim: int, ema: float = 0.95):
        super().__init__()
        self.ema = ema
        # Running estimate of error variance per feature
        self.register_buffer('error_variance', torch.ones(hidden_dim))
        # How stable is our variance estimate? (second-order confidence)
        self.register_buffer('variance_of_variance', torch.ones(hidden_dim))
        # How many updates we've seen
        self.register_buffer('n_updates', torch.tensor(0))

    def update(self, error: torch.Tensor) -> None:
        """
        Update precision estimates from new error observations.

        Args:
            error: [batch, seq_len, hidden_dim] prediction errors
        """
        with torch.no_grad():
            # Per-feature error variance across batch and sequence
            new_var = error.detach().var(dim=(0, 1)).clamp(min=1e-10)

            # Remember old variance for confidence update
            old_var = self.error_variance.clone()

            # Exponential moving average update
            self.error_variance = (
                self.ema * self.error_variance + (1 - self.ema) * new_var
            )

            # Second-order confidence: how much did our estimate change?
            var_delta = (self.error_variance - old_var).abs()
            self.variance_of_variance = (
                0.99 * self.variance_of_variance + 0.01 * var_delta ** 2
            )

            self.n_updates += 1

    def get_precision(self) -> torch.Tensor:
        """
        Get self-normalizing, confidence-modulated precision weights.

        Returns:
            precision: [hidden_dim] where mean ≈ 1.0
        """
        # Raw precision = inverse variance
        raw = 1.0 / (self.error_variance + 1e-8)

        # Confidence: trust precision more when it's stable
        confidence = 1.0 / (self.variance_of_variance + 1e-8)
        confidence = confidence / (confidence.mean() + 1e-8)

        # Modulate precision by confidence
        effective = raw * confidence

        # Self-normalize so mean = 1.0 (homeostatic plasticity)
        return effective / (effective.mean() + 1e-8)


class BasalGangliaGate(nn.Module):
    """
    Error gating: only large, coherent error patterns pass through.

    How it works:
      1. Compress error into a small bottleneck (like squeezing water
         through a narrow pipe — only strong signals get through)
      2. Apply tanh (squashes small values, passes large ones)
      3. Expand back to full size

    Result: 248x selectivity ratio between large and small errors.
    Small/noisy errors are effectively zeroed out. Large, coherent
    errors pass through and drive learning.

    Brain analog: The basal ganglia (a brain structure) selects which
    cortical error signals get to influence learning. Without it
    (Parkinson's disease), everything tries to influence everything,
    leading to rigidity and inability to act.

    Source: Doc 22, Case 4 — basal ganglia gate gave 248x selectivity.

    In pure_local mode, this learns via Hebbian rules.
    In hybrid mode, this learns via backprop through the stress loss.
    """

    def __init__(self, hidden_dim: int, bottleneck_ratio: float = 0.25):
        super().__init__()
        bottleneck = max(16, int(hidden_dim * bottleneck_ratio))
        # PURE PHYSICS: Gate weights are FIXED (buffers, not parameters).
        # The brain's basal ganglia has fixed anatomical structure;
        # its function comes from dopaminergic modulation, not weight changes.
        # The gate's job is just to filter — compress, squash, expand.
        # It doesn't need to learn. W will learn around it.
        down_w = torch.empty(bottleneck, hidden_dim)
        up_w = torch.empty(hidden_dim, bottleneck)
        nn.init.xavier_uniform_(down_w, gain=0.1)
        nn.init.xavier_uniform_(up_w, gain=0.1)
        self.register_buffer('down_weight', down_w)
        self.register_buffer('up_weight', up_w)
        self.bottleneck = bottleneck

    def forward(self, error: torch.Tensor) -> torch.Tensor:
        """
        Gate error signals using fixed random projections.

        Args:
            error: [..., hidden_dim] prediction errors

        Returns:
            gated_error: [..., hidden_dim] filtered (large errors pass)
        """
        compressed = error @ self.down_weight.T   # Compress to bottleneck
        filtered = torch.tanh(compressed)          # Squash small errors
        return filtered @ self.up_weight.T         # Expand back


# ============================================================
# Cross-Frequency Coupling: Note
# ============================================================
#
# CFC is NOT implemented as a separate force or module.
# It is ALREADY present in the tanh force (force 3: nonlinear saturation).
#
# The cubic term in tanh(x) ≈ x - x³/3 creates three-mode coupling
# between eigenmodes. Slow modes (low eigenvalue) modulate fast modes
# (high eigenvalue) through the x³ cross-terms.
#
# With 3 settling steps, CFC contributes ~1% of the total dynamics.
# For stronger CFC, reduce damping ratio or use ACh to lower mass
# (more oscillation cycles in the same settling time).
#
# WARNING: Do NOT add a separate CFC force — this DOUBLE-COUNTS
# the cubic coupling that tanh already provides, and has been
# experimentally confirmed to hurt accuracy and collapse mass
# differentiation.
#
# See SOMI_2_0_CFC_MATHEMATICAL_ANALYSIS.md for the full derivation
# and the CFC-Memory tradeoff analysis.


# ============================================================
# Forces on the field φ
# ============================================================

def compute_field_forces(
    phi: torch.Tensor,          # [batch, seq, hidden]
    phi_dot: torch.Tensor,      # [batch, seq, hidden]
    phi_target: torch.Tensor,   # [batch, seq, hidden]
    W: torch.Tensor,            # [hidden, hidden]
    L_rw: torch.Tensor,         # [hidden, hidden]
    precision: torch.Tensor,    # [hidden]
    beta: float,                # Damping coefficient (from calibration)
    config: SOMIConfig,
    gate: Optional[BasalGangliaGate] = None,
    training: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute ALL forces acting on the field φ.

    Nine forces, each with a clear physical origin and brain analog.
    Forces 1-7 come from the action (conservative: −∇_φ V).
    Forces 8-9 are non-conservative (damping and noise).

    Returns a dict with each force separately (for diagnostics) plus 'total'.
    """
    error = phi - phi_target  # Prediction error

    forces = {}

    # ---- Forces from the action (f = −∇_φ V) ----

    # 1. COUPLING: Activity spreads across connected features
    #    From V_coupling = (α₁/2) φᵀL_rw φ
    #    Brain: synaptic coupling smooths representations
    forces['coupling'] = -config.alpha_1 * (phi @ L_rw.T)

    # 2. ANCHORING: Restoring force toward zero
    #    From V_anchor = (α₀/2)|φ|²
    #    Brain: leak conductance prevents drift
    forces['anchor'] = -config.alpha_0 * phi

    # 3. NONLINEAR SATURATION: Prevents unbounded growth
    #    From V_nonlinear = Σ ln(cosh(φᵢ))
    #    Brain: firing rate saturation (neurons can't fire infinitely fast)
    forces['nonlinear'] = -torch.tanh(phi.clamp(-10, 10))

    # 4. INFO STRESS: Precision-weighted prediction error
    #    From V_info = (κ₀/2) Σ Πᵢ(φᵢ − φ̂ᵢ)²
    #    Brain: cortical columns correct predictions proportional to precision
    prec = precision.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
    forces['info'] = -config.kappa_0 * prec * error

    # 5. BASAL GANGLIA GATE: Filters small/noisy errors
    #    Not directly from the action (external drive)
    #    Brain: basal ganglia selects which errors matter
    if gate is not None and config.kappa_1 > 0:
        forces['gate'] = -config.kappa_1 * gate(error)
    else:
        forces['gate'] = torch.zeros_like(phi)

    # 6. ERROR SMOOTHING (SOMI 2.0 NEW): Errors diffuse to neighbors
    #    From V_error_smooth = (λ_E/2) eᵀL_rw e
    #    Brain: lateral inhibition in cortex shares error responsibility
    forces['error_smooth'] = -config.lambda_E * (error @ L_rw.T)

    # 7. COORDINATION (SOMI 2.0 NEW): Connected features co-activate
    #    From V_coord = −(λ_C/4) φᵀWφ
    #    Brain: recurrent excitation — "fire together, wire together"
    #    Note: positive sign because V_coord is negative in potential
    forces['coordination'] = 0.5 * config.lambda_C * (phi @ W.T)

    # ---- Forces NOT from the action (dissipative/stochastic) ----

    # 8. DAMPING: Energy dissipation
    #    From Rayleigh dissipation R = (1/2) φ̇ᵀ B(W) φ̇
    #    Brain: membrane resistance drains kinetic energy
    #    beta can be a scalar (constant M) or per-feature tensor (geometric inertia)
    if isinstance(beta, (int, float)):
        forces['damping'] = -beta * phi_dot
    else:
        # Per-feature damping: beta is [hidden_dim], broadcast over [batch, seq, hidden]
        forces['damping'] = -beta.unsqueeze(0).unsqueeze(0) * phi_dot

    # 9. NOISE: Exploration
    #    Stochastic drive (Langevin dynamics)
    #    Brain: spontaneous neural activity enables exploration
    if training and config.noise_ratio > 0:
        scale = config.noise_ratio * phi.detach().std().clamp(min=1e-6)
        forces['noise'] = scale * torch.randn_like(phi)
    else:
        forces['noise'] = torch.zeros_like(phi)

    # Total force
    forces['total'] = sum(f for k, f in forces.items() if k != 'total')

    return forces


# ============================================================
# Potential V and Hamiltonian H
# ============================================================

def compute_potential(
    phi: torch.Tensor,
    phi_target: torch.Tensor,
    W: torch.Tensor,
    L_rw: torch.Tensor,
    precision: torch.Tensor,
    config: SOMIConfig,
) -> torch.Tensor:
    """
    Compute total potential V(φ, W) from the SOMI 2.0 action.

    V = V_coupling + V_anchor + V_nonlinear + V_info + V_error_smooth
        + V_coord + V_weight_reg

    Each term corresponds to one line in the action functional.
    Returns a scalar (averaged over batch and sequence).
    """
    error = phi - phi_target
    prec = precision.unsqueeze(0).unsqueeze(0)

    V = torch.tensor(0.0, device=phi.device, dtype=phi.dtype)

    # V_coupling = (α₁/2) φᵀL_rw φ
    V = V + 0.5 * config.alpha_1 * (phi * (phi @ L_rw.T)).sum(dim=-1).mean()

    # V_anchor = (α₀/2)|φ|²
    V = V + 0.5 * config.alpha_0 * (phi ** 2).sum(dim=-1).mean()

    # V_nonlinear = Σ ln(cosh(φᵢ))
    V = V + torch.log(torch.cosh(phi.clamp(-10, 10))).sum(dim=-1).mean()

    # V_info = (κ₀/2) Σ Πᵢ(φᵢ − φ̂ᵢ)²
    V = V + 0.5 * config.kappa_0 * (prec * error ** 2).sum(dim=-1).mean()

    # V_error_smooth = (λ_E/2) eᵀL_rw e
    V = V + 0.5 * config.lambda_E * (error * (error @ L_rw.T)).sum(dim=-1).mean()

    # V_coord = −(λ_C/4) φᵀWφ
    V = V - 0.25 * config.lambda_C * (phi * (phi @ W.T)).sum(dim=-1).mean()

    return V


def compute_hamiltonian(
    phi: torch.Tensor,
    phi_dot: torch.Tensor,
    phi_target: torch.Tensor,
    W: torch.Tensor,
    L_rw: torch.Tensor,
    precision: torch.Tensor,
    config: SOMIConfig,
    M_vector: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the Hamiltonian H = T + V (total energy).

    This is the Lyapunov function for SOMI:
      - dH/dt = -phi_dot^T B(W) phi_dot <= 0  (always decreasing during settling)
      - H decreasing means the system is moving toward equilibrium
      - If H INCREASES, something is broken (see diagnostics.py)

    With geometry-dependent mass (SOMI 2.0 Complete Theory):
      T = (1/2) sum_i M_i(W) * phi_dot_i^2

    Brain analog: The brain minimizes free energy (Friston's Free Energy
    Principle). H is SOMI's version of free energy.

    Args:
        phi, phi_dot, phi_target, W, L_rw, precision, config: Standard SOMI state
        M_vector: [hidden_dim] per-feature mass. If None, uses constant config.M.

    Returns scalar.
    """
    if M_vector is not None:
        # Per-feature kinetic energy: T = (1/2) sum_i M_i * phi_dot_i^2
        # M_vector: [hidden], phi_dot: [batch, seq, hidden]
        M_expanded = M_vector.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        T = 0.5 * (M_expanded * phi_dot ** 2).sum(dim=-1).mean()
    else:
        # Constant mass: T = (M/2)|phi_dot|^2
        T = 0.5 * config.M * (phi_dot ** 2).sum(dim=-1).mean()

    # Potential energy
    V = compute_potential(phi, phi_target, W, L_rw, precision, config)

    return T + V
