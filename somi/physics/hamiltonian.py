"""
SOMI 4.0 Hamiltonian & Stability
==================================

The Hamiltonian H = T + V is SOMI's total energy (Lyapunov function).

    H = (1/2) * sum_i(M_i * phi_dot_i^2) + V(phi, W)

Key principle: dH/dt <= 0 ALWAYS during settling (energy must decrease).
If H ever increases, something is broken — this is SOMI's #1 diagnostic.

Entropy tracking (Vanchurin's Second Law of Learning):
    S = -sum_i(p_i * log(p_i))  where p_i = phi_i^2 / sum(phi^2)
    dS/dt <= 0 during learning (entropy must decrease over training)

Free Energy framework (Friston):
    F = E - T*S  (free energy = energy - temperature * entropy)

CFL condition (Level 3):
    dt < dx / c_info  (information speed limit)
    c_info = sqrt(alpha_1 / (M * dx^2))

Source: SOMI_3_0/theory/24_THE_5_LEVELS_COMPLETE_REFERENCE.md (Levels 1-3)
Reuses: somi_2_0/core.py compute_hamiltonian (base computation)
"""

import torch
import math
from typing import Dict, Optional, Tuple

from .forces import compute_potential


def compute_hamiltonian(
    phi: torch.Tensor,
    phi_dot: torch.Tensor,
    phi_target: torch.Tensor,
    W: torch.Tensor,
    L_rw: torch.Tensor,
    precision: torch.Tensor,
    config: 'SOMIBrainConfig',
    M_vector: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute total energy H = T + V (the Lyapunov function).

    The Hamiltonian is like the total energy of a ball rolling in a bowl:
    - T = kinetic energy (how fast the ball is moving)
    - V = potential energy (how high up the sides of the bowl it is)
    - H = T + V must decrease as the ball settles to the bottom

    If H increases during settling, something is wrong — like a ball
    spontaneously rolling uphill. This NEVER happens in correct physics,
    so it's our most reliable diagnostic.

    Args:
        phi: [batch, seq, hidden] field activity
        phi_dot: [batch, seq, hidden] field velocity
        phi_target: [batch, seq, hidden] target
        W: [hidden, hidden] connectivity
        L_rw: [hidden, hidden] random-walk Laplacian
        precision: [hidden] precision weights
        config: SOMIBrainConfig
        M_vector: [hidden] per-feature mass (None = constant config.M)

    Returns:
        H: scalar (total energy averaged over batch/seq)
        diagnostics: Dict with T, V, H values
    """
    # Kinetic energy T = (1/2) * sum_i(M_i * phi_dot_i^2)
    if M_vector is not None:
        if phi_dot.dim() > M_vector.dim():
            M_expanded = M_vector
            for _ in range(phi_dot.dim() - M_vector.dim()):
                M_expanded = M_expanded.unsqueeze(0)
        else:
            M_expanded = M_vector
        T = 0.5 * (M_expanded * phi_dot ** 2).sum(dim=-1).mean()
    else:
        T = 0.5 * config.M * (phi_dot ** 2).sum(dim=-1).mean()

    # Potential energy V(phi, W)
    V, V_diagnostics = compute_potential(phi, phi_target, W, L_rw, precision, config)

    # Total Hamiltonian
    H = T + V

    diagnostics = {
        'hamiltonian_T': T.item(),
        'hamiltonian_V': V.item(),
        'hamiltonian_H': H.item(),
        **V_diagnostics,
    }

    return H, diagnostics


class HamiltonianTracker:
    """
    Tracks Hamiltonian over time and detects violations.

    SOMI's most important diagnostic: if H increases during settling,
    the physics is broken. This tracker watches H at every settling step
    and raises an alarm if it ever goes up.

    It also tracks:
    - dH/dt (rate of energy change — should be negative)
    - H history (for plotting energy descent)
    - Violation count (how many times H increased)

    Brain analog: The brain's free energy (Friston) must decrease during
    perception and learning. If it increases, the brain is "confused" —
    predictions are getting worse, not better.
    """

    def __init__(self):
        self.H_history = []
        self.dH_history = []
        self.violations = 0
        self.total_steps = 0

    def record(self, H: float) -> Dict[str, float]:
        """
        Record a new Hamiltonian value and check for violations.

        Args:
            H: Current Hamiltonian value

        Returns:
            diagnostics: Dict with dH, violation status
        """
        self.H_history.append(H)
        self.total_steps += 1

        diagnostics = {'hamiltonian_current': H}

        if len(self.H_history) >= 2:
            dH = H - self.H_history[-2]
            self.dH_history.append(dH)
            diagnostics['hamiltonian_dH'] = dH
            diagnostics['hamiltonian_decreasing'] = dH <= 0

            if dH > 1e-6:  # Small tolerance for numerical noise
                self.violations += 1
                diagnostics['hamiltonian_violation'] = True
            else:
                diagnostics['hamiltonian_violation'] = False

        diagnostics['hamiltonian_violations_total'] = self.violations
        diagnostics['hamiltonian_violation_rate'] = (
            self.violations / max(1, self.total_steps - 1)
        )

        return diagnostics

    def reset(self):
        """Reset tracker for a new settling episode."""
        self.H_history = []
        self.dH_history = []


class EntropyTracker:
    """
    Tracks information entropy of the field phi over training.

    Vanchurin's Second Law of Learning (2020):
        dS/dt <= 0 during learning

    This means the model should become MORE organized (lower entropy)
    as it learns. Like a messy room becoming cleaner over time.

    Entropy formula:
        S = -sum_i(p_i * log(p_i))
        where p_i = phi_i^2 / sum(phi^2)

    Each p_i is the fraction of total "energy" in feature i.
    - High entropy = energy spread equally across all features (messy)
    - Low entropy = energy concentrated in a few features (organized)

    A well-trained model should have LOW entropy: it knows which features
    matter for each input and concentrates activity there.
    """

    def __init__(self):
        self.entropy_history = []

    def compute_entropy(self, phi: torch.Tensor) -> float:
        """
        Compute information entropy of the field.

        Args:
            phi: [batch, seq, hidden] or [hidden] field activity

        Returns:
            entropy: scalar (bits of entropy)
        """
        with torch.no_grad():
            # Use squared magnitude as "probability"
            phi_sq = (phi.detach() ** 2).mean(dim=tuple(range(phi.dim() - 1)))  # [hidden]
            total = phi_sq.sum().clamp(min=1e-10)
            p = phi_sq / total  # Probability distribution over features
            p = p.clamp(min=1e-10)  # Avoid log(0)
            entropy = -(p * p.log()).sum().item()

        return entropy

    def record(self, phi: torch.Tensor) -> Dict[str, float]:
        """
        Compute and record entropy.

        Returns:
            diagnostics: Dict with entropy and dS/dt
        """
        S = self.compute_entropy(phi)
        self.entropy_history.append(S)

        diagnostics = {
            'entropy_current': S,
        }

        if len(self.entropy_history) >= 2:
            dS = S - self.entropy_history[-2]
            diagnostics['entropy_dS'] = dS
            diagnostics['entropy_decreasing'] = dS <= 0

        return diagnostics


def compute_free_energy(
    H: float,
    entropy: float,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Compute free energy F = E - T*S (Friston's framework).

    The brain doesn't just minimize energy — it minimizes FREE energy.
    Free energy balances two goals:
    - Low energy E (accurate predictions)
    - High entropy S (keeping options open, not over-committing)

    Temperature T controls the tradeoff:
    - High T = favor entropy (explore, be flexible)
    - Low T = favor energy (exploit, be precise)

    In SOMI, temperature can be tied to arousal or learning phase.

    Args:
        H: Hamiltonian (total energy)
        entropy: Information entropy
        temperature: Temperature parameter

    Returns:
        diagnostics: Dict with F, E, T, S values
    """
    F = H - temperature * entropy

    return {
        'free_energy_F': F,
        'free_energy_E': H,
        'free_energy_T': temperature,
        'free_energy_S': entropy,
    }


def compute_cfl_condition(
    W: torch.Tensor,
    config: 'SOMIBrainConfig',
    M_vector: Optional[torch.Tensor] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the CFL stability condition on dt (Level 3 physics).

    CFL = Courant-Friedrichs-Lewy condition. It's the maximum time step
    that keeps the simulation stable. Think of it like driving:
    - Small dt = driving slowly (always safe, but slow)
    - Large dt = driving fast (efficient, but crash if too fast)
    - CFL dt = the speed limit (fastest safe speed)

    Formula:
        c_info = sqrt(alpha_1 / (M * dx^2))   (information speed)
        dt_safe = dx / c_info_max              (safe time step)
        dt_safe = 1 / sqrt(lambda_max * alpha_1 / M)  (from eigenvalues)

    Where lambda_max is the largest eigenvalue of the Laplacian.

    If config.use_cfl_condition is True, dt should be clamped to dt_safe.

    Args:
        W: [hidden, hidden] connectivity
        config: SOMIBrainConfig
        M_vector: [hidden] per-feature mass (None = constant config.M)

    Returns:
        dt_safe: Maximum stable time step
        diagnostics: CFL metrics
    """
    with torch.no_grad():
        # Estimate spectral radius of L_rw (largest eigenvalue magnitude)
        # Use power iteration for speed (avoid full eigendecomposition)
        n = W.shape[0]
        row_sums = W.sum(dim=1).clamp(min=1e-8)
        D_inv_W = W / row_sums.unsqueeze(1)
        L_rw = torch.eye(n, device=W.device, dtype=W.dtype) - D_inv_W

        # Power iteration: find largest eigenvalue magnitude
        v = torch.randn(n, device=W.device, dtype=W.dtype)
        v = v / v.norm()
        for _ in range(20):
            v_new = L_rw @ v
            eigenvalue_est = v_new.norm()
            v = v_new / (eigenvalue_est + 1e-8)

        lambda_max = eigenvalue_est.item()

        # Information speed: c_info = sqrt(alpha_1 * lambda_max / M)
        M_eff = config.M if M_vector is None else M_vector.mean().item()
        c_info = math.sqrt(max(1e-8, config.alpha_1 * lambda_max / M_eff))

        # CFL condition: dt < 1 / c_info (discrete version)
        dt_safe = 1.0 / max(1e-8, c_info)

        # Safety margin
        dt_safe *= 0.9  # 90% of theoretical limit

        diagnostics = {
            'cfl_c_info': c_info,
            'cfl_dt_safe': dt_safe,
            'cfl_lambda_max': lambda_max,
            'cfl_dt_current': config.dt,
            'cfl_satisfied': config.dt <= dt_safe,
            'cfl_margin': dt_safe / max(1e-8, config.dt),
        }

    return dt_safe, diagnostics


def compute_dynamic_alpha(
    step: int,
    total_steps: int,
    schedule: Tuple[float, float, float] = (1.0, 0.5, 0.0),
) -> float:
    """
    Compute Vanchurin's dynamic alpha for learning regime transitions.

    Three phases:
    1. Exploration (alpha = 1.0): Maximum exploration, geometry explores freely
    2. Efficient learning (alpha = 0.5): Balanced exploration/exploitation
    3. Consolidation (alpha = 0.0): Lock in learned structure, no more exploration

    This is like:
    1. First day at a new job: explore everything, try everything
    2. First month: focus on what works, drop what doesn't
    3. After a year: expert mode, efficient and focused

    From Vanchurin (2020): "The World as a Neural Network" — geometry transitions
    from exploration to exploitation through decreasing alpha.

    Args:
        step: Current training step
        total_steps: Total training steps planned
        schedule: (alpha_phase1, alpha_phase2, alpha_phase3)

    Returns:
        alpha: Current dynamic alpha value
    """
    if total_steps <= 0:
        return schedule[0]

    progress = step / total_steps

    if progress < 0.33:
        return schedule[0]  # Phase 1: exploration
    elif progress < 0.66:
        # Linear interpolation between phase 1 and phase 2
        t = (progress - 0.33) / 0.33
        return schedule[0] * (1 - t) + schedule[1] * t
    else:
        # Linear interpolation between phase 2 and phase 3
        t = (progress - 0.66) / 0.34
        return schedule[1] * (1 - t) + schedule[2] * t
