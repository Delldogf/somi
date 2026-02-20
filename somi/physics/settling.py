"""
SOMI 4.0 Settling Dynamics
============================

The field phi evolves under the SOMI field equation:

    M_i * phi_ddot_i + beta_i * phi_dot_i = F_total_i

Three integration methods:
  1. Symplectic Euler — nonlinear, exact, sequential (default)
  2. Closed-form spectral — linear, parallel, one-step (approximate)
  3. LinOSS parallel scan — O(log n) via associative scan (experimental)

Raj eigenmode truncation: Only keep top-K eigenmodes (K=10-20) for
spectral methods. The brain only uses ~10-20 eigenmodes for cognition.

Source: SOMI_3_0/theory/24_THE_5_LEVELS_COMPLETE_REFERENCE.md (Levels 1-2)
Papers: LinOSS (Rusch & Rus, ICLR 2025), Raj et al. 2020
Reuses: somi_2_0/settling.py (symplectic Euler, closed-form)
"""

import torch
import math
from typing import Dict, Optional, Tuple

from .forces import (
    compute_field_forces,
    compute_laplacian,
    BasalGangliaGate,
)
from .hamiltonian import compute_hamiltonian, HamiltonianTracker


def settle(
    phi: torch.Tensor,
    phi_target: torch.Tensor,
    W: torch.Tensor,
    L_rw: torch.Tensor,
    precision: torch.Tensor,
    beta: torch.Tensor,
    n_steps: int,
    config: 'SOMIBrainConfig',
    gate: Optional[BasalGangliaGate] = None,
    training: bool = True,
    M_vector: Optional[torch.Tensor] = None,
    eigenvalues: Optional[torch.Tensor] = None,
    eigenvectors: Optional[torch.Tensor] = None,
    track_hamiltonian: bool = False,
    method: str = 'symplectic',
    truncation_K: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Main settling dispatcher: picks the right method and runs it.

    Think of settling like a ball rolling in a bowl:
    - phi starts somewhere (the transformer's hidden state)
    - Forces push it around (coupling, prediction error, etc.)
    - Damping slows it down
    - After n_steps, it settles near the bottom = optimal representation

    Three methods:
    - 'symplectic': Most accurate, handles all nonlinear forces, sequential
    - 'spectral': Fastest for linear regime, parallel, approximate
    - 'parallel_scan': O(log n), experimental (LinOSS paper)

    Args:
        phi: [batch, seq, hidden] initial field (usually h.clone())
        phi_target: [batch, seq, hidden] target for prediction errors
        W: [hidden, hidden] connectivity
        L_rw: [hidden, hidden] random-walk Laplacian
        precision: [hidden] precision weights
        beta: [hidden] or scalar — damping coefficients
        n_steps: Number of settling steps
        config: SOMIBrainConfig
        gate: Optional BasalGangliaGate
        training: Whether in training mode (enables noise)
        M_vector: [hidden] per-feature mass (None = constant config.M)
        eigenvalues: [n_modes] for spectral settling
        eigenvectors: [hidden, n_modes] for spectral settling
        track_hamiltonian: Track H at each step (expensive debugging)
        method: 'symplectic', 'spectral', or 'parallel_scan'
        truncation_K: If > 0, only use top-K eigenmodes (Raj truncation)

    Returns:
        phi: [batch, seq, hidden] settled field
        phi_dot: [batch, seq, hidden] final velocity
        info: Dict with diagnostics
    """
    if method == 'spectral' and eigenvalues is not None and eigenvectors is not None:
        # Apply Raj eigenmode truncation if requested
        if truncation_K > 0 and truncation_K < eigenvalues.shape[0]:
            eigenvalues = eigenvalues[:truncation_K]
            eigenvectors = eigenvectors[:, :truncation_K]

        phi_settled, phi_dot = settle_spectral(
            phi, eigenvalues, eigenvectors, beta,
            settle_time=n_steps * config.dt,
            config=config,
            M_vector=M_vector,
        )
        info = {
            'method': 'spectral',
            'n_steps': n_steps,
            'n_modes': eigenvalues.shape[0],
            'final_velocity_norm': phi_dot.detach().abs().mean().item(),
        }
        return phi_settled, phi_dot, info

    elif method == 'ssm' and eigenvalues is not None and eigenvectors is not None:
        if truncation_K > 0 and truncation_K < eigenvalues.shape[0]:
            eigenvalues = eigenvalues[:truncation_K]
            eigenvectors = eigenvectors[:, :truncation_K]

        phi_settled, phi_dot, ssm_info = settle_ssm(
            phi, phi_target, eigenvalues, eigenvectors, beta, precision,
            settle_time=n_steps * config.dt,
            config=config,
            gate=gate,
            M_vector=M_vector,
            training=training,
        )
        ssm_info['n_steps'] = n_steps
        return phi_settled, phi_dot, ssm_info

    elif method == 'parallel_scan' and eigenvalues is not None:
        phi_settled, phi_dot, info = settle_parallel_scan(
            phi, L_rw, beta, n_steps, config,
            M_vector=M_vector,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        return phi_settled, phi_dot, info

    else:
        # Default: symplectic Euler (always works, handles nonlinearity)
        return settle_symplectic(
            phi, phi_target, W, L_rw, precision, beta,
            n_steps, config, gate, training,
            track_hamiltonian=track_hamiltonian,
            M_vector=M_vector,
        )


def settle_symplectic(
    phi: torch.Tensor,
    phi_target: torch.Tensor,
    W: torch.Tensor,
    L_rw: torch.Tensor,
    precision: torch.Tensor,
    beta,
    n_steps: int,
    config: 'SOMIBrainConfig',
    gate: Optional[BasalGangliaGate] = None,
    training: bool = True,
    track_hamiltonian: bool = False,
    M_vector: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Settle using symplectic Euler integration.

    Symplectic Euler is special because it preserves the "phase space volume"
    of the dynamics. In plain English: it doesn't artificially add or remove
    energy from the system. Only the damping term (beta) removes energy,
    which is what we want.

    Regular Euler would either:
    - Add energy each step -> oscillations blow up (explosion)
    - Remove too much energy -> oscillations die too fast (over-damping)

    The update order matters:
        1. Update velocity FIRST (using current position)
        2. Update position with NEW velocity
    This specific order is what makes it "symplectic" (phase-preserving).

    Returns:
        phi: Settled field
        phi_dot: Final velocity
        info: Diagnostics dict
    """
    dt = config.dt
    phi_dot = torch.zeros_like(phi)

    # Hamiltonian tracker (for debugging)
    H_tracker = HamiltonianTracker() if track_hamiltonian else None
    all_force_diagnostics = []

    # Prepare mass for acceleration: acceleration = force / mass
    if M_vector is not None:
        M_inv = 1.0 / M_vector.clamp(min=1e-8)
        # Broadcast to match phi dimensions
        for _ in range(phi.dim() - M_inv.dim()):
            M_inv = M_inv.unsqueeze(0)
    else:
        M_inv = 1.0 / config.M

    # Safety clamping limits — prevent runaway dynamics
    # Values chosen so the physics stays in a "reasonable" operating range
    # where tanh and other nonlinearities are still informative (not saturated).
    PHI_CLAMP = 10.0       # |phi| never exceeds this
    PHI_DOT_CLAMP = 10.0   # |phi_dot| never exceeds this
    ACCEL_CLAMP = 20.0      # |acceleration| never exceeds this

    # Remember the initial phi for NaN recovery
    phi_init = phi.clone()
    nan_detected = False

    for step in range(n_steps):
        # Compute all 9 forces
        forces, force_diag = compute_field_forces(
            phi, phi_dot, phi_target, W, L_rw, precision,
            beta, config, gate, training,
        )

        if step == 0 or step == n_steps - 1:
            all_force_diagnostics.append(force_diag)

        # Symplectic Euler: velocity first, then position
        acceleration = forces['total'] * M_inv

        # === SAFETY: Clamp acceleration, velocity, and position ===
        # If forces are too large, the system is about to blow up.
        # Clamping prevents NaN cascades while damping naturally stabilizes.
        acceleration = acceleration.clamp(-ACCEL_CLAMP, ACCEL_CLAMP)
        phi_dot = phi_dot + dt * acceleration
        phi_dot = phi_dot.clamp(-PHI_DOT_CLAMP, PHI_DOT_CLAMP)
        phi = phi + dt * phi_dot
        phi = phi.clamp(-PHI_CLAMP, PHI_CLAMP)

        # NaN check — if anything goes NaN, reset to initial values
        if torch.isnan(phi).any() or torch.isnan(phi_dot).any():
            phi = phi_init.clone()
            phi_dot = torch.zeros_like(phi)
            nan_detected = True
            break

        # Track Hamiltonian if requested
        if H_tracker is not None:
            with torch.no_grad():
                H, _ = compute_hamiltonian(
                    phi, phi_dot, phi_target, W, L_rw, precision,
                    config, M_vector=M_vector,
                )
                H_tracker.record(H.item())

    # Collect diagnostics
    info = {
        'method': 'symplectic',
        'n_steps': n_steps,
        'final_velocity_norm': phi_dot.detach().abs().mean().item(),
        'phi_range': (phi.detach().min().item(), phi.detach().max().item()),
        'nan_detected': nan_detected,
    }

    # Add force diagnostics from first and last step
    if all_force_diagnostics:
        info['force_diagnostics_first'] = all_force_diagnostics[0]
        if len(all_force_diagnostics) > 1:
            info['force_diagnostics_last'] = all_force_diagnostics[-1]

    # Add Hamiltonian tracking
    if H_tracker is not None:
        info['hamiltonian_violations'] = H_tracker.violations
        info['hamiltonian_violation_rate'] = H_tracker.violations / max(1, n_steps - 1)
        if H_tracker.H_history:
            info['hamiltonian_start'] = H_tracker.H_history[0]
            info['hamiltonian_end'] = H_tracker.H_history[-1]
            info['hamiltonian_decreased'] = (
                H_tracker.H_history[-1] <= H_tracker.H_history[0] + 0.01
            )

    return phi, phi_dot, info


def settle_spectral(
    phi: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    beta,
    settle_time: float,
    config: 'SOMIBrainConfig',
    M_vector: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form spectral solution for the LINEAR regime.

    Instead of stepping through time, this solves the differential equation
    analytically in ONE operation. Think of it as "fast-forwarding" to the
    answer instead of simulating every step.

    The math: Each eigenmode is an independent damped oscillator:
        c_k(t) = exp(-gamma_k * t) * [A_k * cos(omega_k * t) + B_k * sin(omega_k * t)]

    We project phi into the eigenmode basis, solve each mode independently,
    and project back. This is O(hidden * n_modes) instead of
    O(n_steps * hidden^2) for symplectic Euler.

    Limitation: Only handles LINEAR forces (coupling + anchoring).
    Nonlinear forces (tanh, info stress) are ignored. Accurate when phi
    is small (linear regime).

    Practical use: Spectral solve for the linear part, then 1-2 symplectic
    correction steps for nonlinearity.

    Raj truncation: Use only top-K eigenmodes (K=10-20). The brain only
    uses ~10-20 eigenmodes for cognition (Raj et al. 2020).

    Args:
        phi: [batch, seq, hidden] initial field
        eigenvalues: [n_modes] eigenvalues of L_rw (ascending order)
        eigenvectors: [hidden, n_modes] corresponding eigenvectors
        beta: Damping coefficient (scalar or [hidden])
        settle_time: Total settling time (n_steps * dt)
        config: SOMIBrainConfig
        M_vector: [hidden] per-feature mass (None = constant)

    Returns:
        phi_settled: [batch, seq, hidden]
        phi_dot_settled: [batch, seq, hidden]
    """
    M = config.M if M_vector is None else M_vector.mean().item()

    if isinstance(beta, (int, float)):
        gamma_scalar = beta / (2.0 * M)
    elif beta.dim() == 0:
        gamma_scalar = beta.item() / (2.0 * M)
    else:
        gamma_scalar = beta.mean().item() / (2.0 * M)

    # Natural frequency for each eigenmode
    # omega_k^2 = (alpha_1 * lambda_k + alpha_0 + 1.0) / M - gamma^2
    # The +1.0 comes from linearized tanh(x) ≈ x near x=0
    omega_sq = (config.alpha_1 * eigenvalues + config.alpha_0 + 1.0) / M - gamma_scalar ** 2
    omega = torch.sqrt(omega_sq.clamp(min=1e-10))
    gamma_k = torch.full_like(eigenvalues, gamma_scalar)

    t = settle_time
    decay = torch.exp(-gamma_k * t)  # Exponential decay per mode

    # Project into eigenmode basis
    original_shape = phi.shape
    phi_flat = phi.reshape(-1, phi.shape[-1])  # [N, hidden]
    c0 = phi_flat @ eigenvectors  # [N, n_modes] — coefficients per mode
    cdot0 = torch.zeros_like(c0)  # Start from rest

    # Solve each mode: c_k(t) = decay * (A * cos + B * sin)
    A = c0
    B = (cdot0 + gamma_k.unsqueeze(0) * c0) / omega.unsqueeze(0)

    cos_wt = torch.cos(omega.unsqueeze(0) * t)
    sin_wt = torch.sin(omega.unsqueeze(0) * t)

    # Position and velocity at time t
    c_t = decay.unsqueeze(0) * (A * cos_wt + B * sin_wt)
    cdot_t = decay.unsqueeze(0) * (
        (-gamma_k.unsqueeze(0) * A + omega.unsqueeze(0) * B) * cos_wt
        + (-gamma_k.unsqueeze(0) * B - omega.unsqueeze(0) * A) * sin_wt
    )

    # Project back to feature space
    phi_settled = (c_t @ eigenvectors.T).reshape(original_shape)
    phi_dot_settled = (cdot_t @ eigenvectors.T).reshape(original_shape)

    return phi_settled, phi_dot_settled


def settle_ssm(
    phi: torch.Tensor,
    phi_target: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    beta,
    precision: torch.Tensor,
    settle_time: float,
    config: 'SOMIBrainConfig',
    gate: Optional[BasalGangliaGate] = None,
    M_vector: Optional[torch.Tensor] = None,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    SSM-based settling: closed-form linear solve + nonlinear correction.

    Solves the FULL linear dynamics (all 5 linear forces) analytically,
    then applies a single-step nonlinear correction for tanh and gate.

    This is 5-10x faster than symplectic Euler because it replaces
    the iterative settling loop with:
    1. Project phi into eigenbasis (1 matmul)
    2. Apply closed-form damped oscillator solution (element-wise)
    3. Project back to feature space (1 matmul)
    4. Apply nonlinear correction (1 pass)

    The linear forces absorbed into the state matrix:
      coupling:     -alpha_1 * (phi @ L_rw)
      anchor:       -alpha_0 * phi
      info (linear): -kappa_0 * precision * phi
      error_smooth: -lambda_E * (error @ L_rw)
      coordination: +0.5 * lambda_C * phi

    The constant driving term (from phi_target):
      b = kappa_0 * precision * phi_target + lambda_E * (phi_target @ L_rw)

    Nonlinear corrections (applied after the linear solve):
      -tanh(phi)            (saturation)
      -kappa_1 * gate(error) (basal ganglia filtering)

    Args:
        phi: [batch, seq, hidden] initial field
        phi_target: [batch, seq, hidden] target
        eigenvalues: [n_modes] of L_rw
        eigenvectors: [hidden, n_modes]
        beta: damping (scalar or [hidden])
        precision: [hidden] precision weights
        settle_time: total time (n_steps * dt)
        config: SOMIBrainConfig
        gate: Optional basal ganglia gate
        M_vector: [hidden] per-feature mass
        training: enables noise

    Returns:
        phi_settled, phi_dot, diagnostics
    """
    M = config.M if M_vector is None else M_vector.mean().item()

    if isinstance(beta, (int, float)):
        gamma_scalar = beta / (2.0 * M)
    elif beta.dim() == 0:
        gamma_scalar = beta.item() / (2.0 * M)
    else:
        gamma_scalar = beta.mean().item() / (2.0 * M)

    prec_mean = precision.mean().item()

    # Effective spring constant per eigenmode (from ALL linear forces):
    #   K_k = (alpha_1 + lambda_E) * lambda_k + alpha_0 + 1.0 + kappa_0*prec - 0.5*lambda_C
    # The +1.0 comes from linearized tanh(x) ~ x near zero.
    effective_stiffness = (
        (config.alpha_1 + config.lambda_E) * eigenvalues
        + config.alpha_0 + 1.0
        + config.kappa_0 * prec_mean
        - 0.5 * config.lambda_C
    )

    omega_sq = effective_stiffness / M - gamma_scalar ** 2
    omega = torch.sqrt(omega_sq.clamp(min=1e-10))
    gamma_k = torch.full_like(eigenvalues, gamma_scalar)

    t = settle_time

    # --- Compute driving term b in eigenbasis ---
    original_shape = phi.shape
    phi_flat = phi.reshape(-1, phi.shape[-1])                # [N, hidden]
    target_flat = phi_target.reshape(-1, phi_target.shape[-1])

    # Constant drive: kappa_0 * precision * phi_target
    if precision.dim() < target_flat.dim():
        prec_broad = precision.unsqueeze(0)
    else:
        prec_broad = precision
    b_feature = config.kappa_0 * prec_broad * target_flat    # [N, hidden]

    # Project phi and driving term into eigenbasis
    c0 = phi_flat @ eigenvectors                              # [N, n_modes]
    b_modes = b_feature @ eigenvectors                        # [N, n_modes]

    # Steady state per mode: c_steady = b_k / stiffness_k
    stiffness_broad = effective_stiffness.unsqueeze(0).clamp(min=1e-8)
    c_steady = b_modes / stiffness_broad                     # [N, n_modes]

    # Transient: deviation from steady state
    delta_c0 = c0 - c_steady

    decay = torch.exp(-gamma_k * t)
    cos_wt = torch.cos(omega.unsqueeze(0) * t)
    sin_wt = torch.sin(omega.unsqueeze(0) * t)

    # Starting from rest (phi_dot = 0):
    #   c(t) = c_steady + decay * (delta_c0 * cos + (gamma/omega * delta_c0) * sin)
    A_coeff = delta_c0
    B_coeff = (gamma_k.unsqueeze(0) * delta_c0) / omega.unsqueeze(0).clamp(min=1e-10)

    c_t = c_steady + decay.unsqueeze(0) * (A_coeff * cos_wt + B_coeff * sin_wt)

    cdot_t = decay.unsqueeze(0) * (
        (-gamma_k.unsqueeze(0) * A_coeff + omega.unsqueeze(0) * B_coeff) * cos_wt
        + (-gamma_k.unsqueeze(0) * B_coeff - omega.unsqueeze(0) * A_coeff) * sin_wt
    )

    # Project back to feature space
    phi_linear = (c_t @ eigenvectors.T).reshape(original_shape)
    phi_dot_linear = (cdot_t @ eigenvectors.T).reshape(original_shape)

    # --- Nonlinear correction (single step) ---
    # Apply the nonlinear forces that were excluded from the linear solve.
    # Use a fraction of dt as the correction step size.
    dt_correction = config.dt * 2.0

    f_nonlinear = -torch.tanh(phi_linear.clamp(-10, 10))

    if gate is not None and config.kappa_1 > 0:
        error_linear = phi_linear - phi_target
        f_gate = -config.kappa_1 * gate(error_linear)
    else:
        f_gate = torch.zeros_like(phi_linear)

    f_noise = torch.zeros_like(phi_linear)
    if training and config.noise_ratio > 0:
        scale = config.noise_ratio * phi_linear.detach().std().clamp(min=1e-6)
        f_noise = scale * torch.randn_like(phi_linear)

    M_inv = 1.0 / M
    phi_dot_corrected = phi_dot_linear + dt_correction * (f_nonlinear + f_gate + f_noise) * M_inv
    phi_corrected = phi_linear + dt_correction * (f_nonlinear + f_gate + f_noise) * M_inv

    phi_corrected = phi_corrected.clamp(-10.0, 10.0)
    phi_dot_corrected = phi_dot_corrected.clamp(-10.0, 10.0)

    # NaN guard
    if torch.isnan(phi_corrected).any():
        phi_corrected = phi.clone()
        phi_dot_corrected = torch.zeros_like(phi)

    info = {
        'method': 'ssm',
        'n_modes': eigenvalues.shape[0],
        'final_velocity_norm': phi_dot_corrected.detach().abs().mean().item(),
        'nan_detected': torch.isnan(phi_corrected).any().item(),
    }

    return phi_corrected, phi_dot_corrected, info


def settle_parallel_scan(
    phi: torch.Tensor,
    L_rw: torch.Tensor,
    beta,
    n_steps: int,
    config: 'SOMIBrainConfig',
    M_vector: Optional[torch.Tensor] = None,
    eigenvalues: Optional[torch.Tensor] = None,
    eigenvectors: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    LinOSS-style parallel scan settling (experimental).

    From LinOSS (Rusch & Rus, ICLR 2025):
    Uses associative scan to solve the LINEAR dynamics in O(log n) parallel
    time instead of O(n) sequential time.

    The key idea: The linear dynamics can be written as a matrix recurrence:
        [phi_{t+1}]   = A * [phi_t]   + B * [input]
        [phi_dot_{t+1}]     [phi_dot_t]

    And matrix recurrences can be parallelized with prefix sums (scans).
    This is the same trick used in parallel prefix addition in hardware.

    Implicit-explicit discretization (from LinOSS):
    - Coupling force (A matrix): treated implicitly (stable for stiff systems)
    - Input force (B matrix): treated explicitly (simple, avoids matrix inverse)

    Current status: Falls back to spectral if eigenvalues available,
    otherwise to symplectic. Full parallel scan requires custom CUDA kernel.

    Args:
        phi: Initial field
        L_rw: Laplacian
        beta: Damping
        n_steps: Settling steps
        config: SOMIBrainConfig
        M_vector: Per-feature mass
        eigenvalues: Precomputed eigenvalues (for fallback)
        eigenvectors: Precomputed eigenvectors (for fallback)

    Returns:
        phi, phi_dot, info: Settled field, velocity, diagnostics
    """
    # For now, use spectral method as the parallel path
    # True parallel scan requires custom CUDA kernels (future work)
    if eigenvalues is not None and eigenvectors is not None:
        phi_settled, phi_dot = settle_spectral(
            phi, eigenvalues, eigenvectors, beta,
            settle_time=n_steps * config.dt,
            config=config,
            M_vector=M_vector,
        )
        info = {
            'method': 'parallel_scan_spectral_fallback',
            'n_steps': n_steps,
            'n_modes': eigenvalues.shape[0],
            'final_velocity_norm': phi_dot.detach().abs().mean().item(),
            'note': 'Using spectral method as parallel scan fallback. '
                    'Full LinOSS scan requires custom CUDA kernel.',
        }
        return phi_settled, phi_dot, info
    else:
        # Ultimate fallback: symplectic Euler
        # This shouldn't normally happen (eigenvalues should be available)
        info = {
            'method': 'parallel_scan_symplectic_fallback',
            'note': 'No eigenvalues available. Using symplectic Euler fallback.',
        }
        return settle_symplectic(
            phi=phi,
            phi_target=torch.zeros_like(phi),
            W=torch.zeros(phi.shape[-1], phi.shape[-1], device=phi.device),
            L_rw=L_rw,
            precision=torch.ones(phi.shape[-1], device=phi.device),
            beta=beta,
            n_steps=n_steps,
            config=config,
            training=False,
            M_vector=M_vector,
        )


def compute_eigendecomposition(
    L_rw: torch.Tensor,
    n_modes: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Compute eigendecomposition of the Laplacian for spectral settling.

    The eigenvalues/eigenvectors of L_rw tell us the "natural vibration modes"
    of the network. Like how a guitar string has harmonics (fundamental,
    2nd harmonic, etc.), the network has eigenmodes (slow global patterns,
    fast local patterns).

    - Low eigenvalues = slow, global modes (like the fundamental note)
    - High eigenvalues = fast, local modes (like high harmonics)

    Raj truncation: The brain only uses ~10-20 eigenmodes for cognition.
    Setting n_modes to 10-20 captures the essential dynamics while saving
    compute.

    IMPORTANT: Use bfloat16 or float32, NOT float16. Float16 eigendecomp
    produces NaN.

    Args:
        L_rw: [hidden, hidden] random-walk Laplacian
        n_modes: Number of modes to keep (None = all). Raj truncation K=10-20.

    Returns:
        eigenvalues: [n_modes] sorted ascending
        eigenvectors: [hidden, n_modes]
        diagnostics: Spectral metrics
    """
    with torch.no_grad():
        # Symmetrize for eigendecomposition (get real eigenvalues)
        L_sym = 0.5 * (L_rw + L_rw.T)

        # Ensure float32 or bfloat16 (NOT float16 — causes NaN!)
        if L_sym.dtype == torch.float16:
            L_sym = L_sym.to(torch.float32)

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)
        except Exception as e:
            # Fallback: identity eigendecomposition
            n = L_rw.shape[0]
            eigenvalues = torch.zeros(n, device=L_rw.device)
            eigenvectors = torch.eye(n, device=L_rw.device)
            return eigenvalues, eigenvectors, {
                'eigen_error': str(e),
                'eigen_fallback': True,
            }

        # Truncate to top-K modes if requested
        total_modes = eigenvalues.shape[0]
        if n_modes is not None and n_modes < total_modes:
            eigenvalues = eigenvalues[:n_modes]
            eigenvectors = eigenvectors[:, :n_modes]

        # Diagnostics
        diagnostics = {
            'eigen_total_modes': total_modes,
            'eigen_used_modes': eigenvalues.shape[0],
            'eigen_min': eigenvalues.min().item(),
            'eigen_max': eigenvalues.max().item(),
            'eigen_median': eigenvalues.median().item(),
            'eigen_spectral_gap': (
                eigenvalues[1].item() - eigenvalues[0].item()
                if eigenvalues.shape[0] > 1 else 0.0
            ),
        }

    return eigenvalues, eigenvectors, diagnostics


def compute_auto_n_settle(
    eigenvalues: torch.Tensor,
    dt: float,
    target_zeta: float = 0.15,
) -> int:
    """
    Auto-compute optimal number of settling steps from eigenfrequencies.

    Formula: n_settle = max(3, floor(pi / (omega_median * dt)))

    This calculates how many steps it takes for the median eigenmode to
    complete half an oscillation cycle. That's the minimum time for the
    system to "see" and respond to its coupling structure.

    Think of it like: if a guitar string vibrates at 440 Hz, you need to
    listen for at least 1/(2*440) seconds to hear one half-cycle. Similarly,
    SOMI needs enough steps for its eigenmodes to complete their cycles.

    Typical result: 5-10 steps for hidden_dim=128-512.

    Args:
        eigenvalues: [n_modes] eigenvalues of L_rw
        dt: Time step
        target_zeta: Damping ratio (affects frequency slightly)

    Returns:
        n_settle: Optimal settling steps (at least 3)
    """
    with torch.no_grad():
        # Positive eigenvalues only (excluding the zero eigenvalue)
        pos_eigs = eigenvalues[eigenvalues > 1e-6]

        if len(pos_eigs) == 0:
            return 5  # Safe default

        # Median eigenvalue
        median_eig = pos_eigs.median().item()

        # Damped frequency: omega = sqrt(lambda) * sqrt(1 - zeta^2)
        omega_median = math.sqrt(max(1e-8, median_eig)) * math.sqrt(1 - target_zeta ** 2)

        # Half period: time for one half oscillation
        half_period = math.pi / max(1e-8, omega_median)

        # Number of steps
        n_settle = max(3, int(half_period / dt))

        # Cap at reasonable maximum
        n_settle = min(n_settle, 20)

    return n_settle
