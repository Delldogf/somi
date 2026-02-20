"""
SOMI 2.0 Self-Calibration
===========================

Every parameter that ISN'T a regime choice is self-calibrated from
the model's internal state. Zero hyperparameters to tune.

Self-calibrating mechanisms:
  1. beta (damping)       ← from target_zeta and eigenfrequencies
  2. n_settle             ← from half-period of median eigenfrequency
  3. eta (geometry LR)    ← from timescale_ratio and arousal
  4. stress_weight        ← from stress ratio to theoretical optimum
  5. arousal              ← from surprise (error vs running average)

The brain does all of this automatically via homeostatic plasticity
and neuromodulation. SOMI does the same.

Source: Doc 24 (Zero Hyperparameters — Self-Calibrating SOMI at LLM Scale)
"""

import math
from ..config import SOMIBrainConfig as SOMIConfig

try:
    import torch
except ImportError:
    pass  # Allow module to be imported for docs without torch


# ============================================================
# Geometry-Dependent Mass (SOMI 2.0 Complete Theory)
# ============================================================

def compute_mass_vector(W, M_0: float):
    """
    Compute per-feature inertial mass from geometry W.

    This is NOT an ad hoc extension — it is what you get from proper
    discretization of the continuous SOMI 2.0 action, where the metric
    tensor g^{tt} varies in space. The local connectivity structure
    (encoded in W) determines the effective mass at each node.

    The mass is derived from the Herfindahl index (connection concentration),
    which is the exponential of the negative Renyi 2-entropy of the
    connection distribution — an information-geometric quantity.

    Formula:
        h_i = sum_j W_ij^2           (Herfindahl index = connection concentration)
        M_i = M_0 * h_mean / h_i     (inverse concentration = mass)

    Properties:
        - Concentrated connections (few strong) = LOW mass = FAST oscillation
          (like well-myelinated brain circuits)
        - Flat connections (many weak) = HIGH mass = SLOW oscillation
          (like weakly connected brain regions)
        - Self-normalizing: mean mass is approximately M_0
        - No new parameters: mass is entirely determined by W
        - Bounded: for doubly stochastic W, h_i in [1/N, 1]

    Brain analog: Membrane capacitance varies across cortical regions.
    Well-structured circuits (V1, motor cortex) respond fast. Weakly
    connected regions (association cortex) respond more slowly.

    Validated by: Effenberger et al. (PNAS 2025, HORNs) — heterogeneous
    frequencies improve performance in damped harmonic oscillator RNNs.

    Args:
        W:   [hidden_dim, hidden_dim] connectivity matrix (doubly stochastic)
        M_0: Base mass (regime choice, e.g. 1.0)

    Returns:
        M_vector: [hidden_dim] per-feature mass
    """
    # Herfindahl index: connection concentration per feature
    h = (W ** 2).sum(dim=-1)  # [hidden_dim]
    h_mean = h.mean()

    # Per-feature mass: inverse of relative concentration
    M_vector = M_0 * h_mean / (h + 1e-8)

    # Clamp mass ratio for numerical safety
    # This prevents extreme frequency ratios that would need very small dt
    M_vector = M_vector.clamp(min=M_0 * 0.1, max=M_0 * 10.0)

    return M_vector


def compute_beta_vector(M_vector, K_ii: float, target_zeta: float):
    """
    Compute per-feature damping from per-feature mass.

    Each feature gets the same DAMPING RATIO zeta but different absolute
    damping. This is critical: it means every feature settles at the same
    RATE (relative to its natural frequency) even though they oscillate at
    different frequencies.

    Formula:
        beta_i = 2 * zeta * sqrt(M_i * K_ii)

    where K_ii = alpha_1 + alpha_0 + 1.0 is the local stiffness.

    Brain analog: Membrane resistance scales with capacitance so the
    time constant tau = RC stays proportional across regions.

    Args:
        M_vector:    [hidden_dim] per-feature mass
        K_ii:        Local stiffness (alpha_1 + alpha_0 + 1.0)
        target_zeta: Target damping ratio (regime choice)

    Returns:
        beta_vector: [hidden_dim] per-feature damping coefficient
    """
    return 2.0 * target_zeta * torch.sqrt(M_vector * K_ii)


def compute_herfindahl(W):
    """
    Compute the Herfindahl index (connection concentration) for each feature.

    This is also exp(-H_2) where H_2 is the Renyi 2-entropy of each
    row of W viewed as a probability distribution.

    Args:
        W: [hidden_dim, hidden_dim] connectivity matrix

    Returns:
        h: [hidden_dim] Herfindahl index per feature
    """
    return (W ** 2).sum(dim=-1)


# ============================================================
# Dale's Law: Derived Parameters (Phase 3)
# ============================================================

def derive_dale_parameters(alpha_1: float, ei_ratio: float):
    """
    Derive lambda_E and lambda_C from alpha_1 and the E/I ratio.

    THIS IS THE KEY INSIGHT OF PHASE 3: Dale's Law eliminates two
    free parameters by deriving them from the coupling strength
    and the excitatory/inhibitory ratio.

    The derivation:
      In a network with fraction f_E excitatory and f_I = 1-f_E inhibitory
      features, the coupling potential V_coupling = (alpha_1/2) phi^T L phi
      already captures the AVERAGE coupling through the Laplacian.

      The CORRECTIONS beyond the average arise from the E/I asymmetry:
      - V_coord (coordination): excitatory connections create positive
        feedback (co-activation). Strength proportional to the E-I coupling
        variance = f_E * f_I (maximized at 50/50, zero at 100/0 or 0/100).
      - V_error_smooth (error smoothing): inhibitory connections diffuse
        prediction errors. Same variance scaling = f_E * f_I.

      Formula:
        lambda_C = alpha_1 * f_E * f_I
        lambda_E = alpha_1 * f_E * f_I

      The corrections are EQUAL because E/I balance requires symmetric
      correction strengths. (Individual inhibitory neurons are stronger
      per connection, but there are fewer of them — it balances out.)

    Properties:
      - Both are proportional to coupling strength alpha_1 (no independent scales)
      - Both scale with E/I variance (disappear for all-E or all-I networks)
      - At the brain's 80/20 ratio: lambda_C = lambda_E = 0.16 * alpha_1
      - This is close to the manually-chosen value of 0.1 — the brain got it right!
      - Total correction = 2 * alpha_1 * f_E * f_I (bounded by alpha_1/2)

    Brain analog:
      Dale's Law constrains real neural networks. The error smoothing
      (lateral inhibition) and coordination (recurrent excitation) are
      consequences of the E/I structure, not independent design choices.
      The brain doesn't tune them separately — they emerge from the
      same structural property.

    Args:
        alpha_1:  Coupling strength (regime choice)
        ei_ratio: Fraction of excitatory features (regime choice)

    Returns:
        lambda_E: Error smoothing strength (derived)
        lambda_C: Coordination strength (derived)
    """
    f_E = ei_ratio
    f_I = 1.0 - ei_ratio

    # The correction strength is the E/I variance
    correction = alpha_1 * f_E * f_I

    lambda_C = correction  # Coordination from E connections
    lambda_E = correction  # Error smoothing from I connections

    return lambda_E, lambda_C


def compute_beta(
    eigenvalues,  # torch.Tensor or None
    config: SOMIConfig,
) -> float:
    """
    Compute damping coefficient beta from target zeta and eigenfrequencies.

    Formula:
        β = 2 × ζ × ω_med × M

    where:
        ζ (zeta)   = target damping ratio (regime choice, e.g. 0.15)
        ω_med      = median natural frequency of the network
        M          = inertial mass

    The median frequency comes from:
        ω_med = √((α₁ λ_med + α₀ + 1.0) / M)

    The +1.0 comes from linearizing tanh(φ) ≈ φ near zero,
    which adds an effective spring constant of 1.0.

    Brain analog: Membrane time constant (τ = RC) determines how quickly
    neural oscillations decay. This isn't a tuning knob — it's a physical
    property that emerges from membrane capacitance and resistance.

    Args:
        eigenvalues: Eigenvalues of L_rw, or None (uses default λ_med=1.0)
        config: SOMI configuration

    Returns:
        beta: Damping coefficient (float)
    """
    # Median eigenvalue (default 1.0 for flat/initial W)
    if eigenvalues is None or (hasattr(eigenvalues, '__len__') and len(eigenvalues) == 0):
        lambda_med = 1.0
    else:
        lambda_med = float(eigenvalues.median().item())

    # Median natural frequency (including tanh linearization)
    omega_med = math.sqrt(
        (config.alpha_1 * lambda_med + config.alpha_0 + 1.0) / config.M
    )

    # Damping coefficient
    beta = 2.0 * config.target_zeta * omega_med * config.M
    return beta


def compute_n_settle(
    eigenvalues,  # torch.Tensor or None
    config: SOMIConfig,
) -> int:
    """
    Auto-compute number of settling steps from eigenfrequencies.

    Formula:
        n_settle = max(3, ⌊π / (ω_med × dt)⌋)

    This is the half-period of the median eigenfrequency — enough time
    for one complete oscillation cycle to explore the energy landscape
    and return toward equilibrium.

    Brain analog: The time it takes a cortical column to complete one
    processing cycle. Emerges from membrane properties, not chosen by
    the brain. Typically ~10-20ms for gamma-band oscillations.

    With default parameters:
        λ_med ≈ 1.0, ω_med ≈ 1.449, dt = 0.15 → n_settle ≈ 14

    Args:
        eigenvalues: Eigenvalues of L_rw, or None
        config: SOMI configuration

    Returns:
        n_settle: Number of settling steps (int, ≥ 3)
    """
    # If user explicitly set n_settle, use it
    if config.n_settle > 0:
        return config.n_settle

    # Median eigenvalue
    if eigenvalues is None or (hasattr(eigenvalues, '__len__') and len(eigenvalues) == 0):
        lambda_med = 1.0
    else:
        lambda_med = float(eigenvalues.median().item())

    # Median natural frequency
    omega_med = math.sqrt(
        (config.alpha_1 * lambda_med + config.alpha_0 + 1.0) / config.M
    )

    # Half-period
    n = max(3, int(math.pi / (omega_med * config.dt)))
    return n


def compute_eta(
    arousal: float,
    config: SOMIConfig,
) -> float:
    """
    Compute geometry learning rate from arousal and timescale ratio.

    Formula:
        η_base = 0.1 / timescale_ratio
        η = η_base × (0.5 + arousal)

    When arousal is high (the model is surprised), geometry learns faster.
    When arousal is low (everything is familiar), geometry consolidates.

    Brain analog: Synaptic plasticity rate is modulated by neuromodulators.
    - Norepinephrine (from locus coeruleus) increases with surprise
    - High norepinephrine → more plastic → learn geometry faster
    - Low norepinephrine → consolidate → geometry stabilizes

    Args:
        arousal: Current arousal level (0 to 1, from compute_arousal)
        config: SOMI configuration

    Returns:
        eta: Geometry learning rate (float)
    """
    eta_base = 0.1 / config.timescale_ratio
    return eta_base * (0.5 + arousal)


def compute_arousal(
    error_magnitude: float,
    running_avg: float,
) -> float:
    """
    Compute arousal from surprise (error relative to expectation).

    Formula:
        relative_surprise = error / running_avg − 1
        arousal = sigmoid(relative_surprise)

    If the current error is HIGHER than expected → high arousal → learn more.
    If the current error is LOWER than expected → low arousal → consolidate.

    Brain analog: Locus coeruleus — norepinephrine system.
    When the brain encounters something unexpected (high surprise),
    norepinephrine spikes → increased plasticity, attention, and arousal.
    When everything is predictable → low arousal → consolidate and rest.

    Args:
        error_magnitude: Current mean absolute error
        running_avg: Running average of past error magnitudes

    Returns:
        arousal: Value between 0.01 and 0.99
    """
    if running_avg < 1e-8:
        return 0.5  # No data yet — neutral arousal

    relative_surprise = error_magnitude / running_avg - 1.0

    # Sigmoid squashes into (0, 1)
    arousal = 1.0 / (1.0 + math.exp(-relative_surprise))

    # Clamp to prevent extreme values
    return max(0.01, min(0.99, arousal))


def compute_stress_weight(
    current_stress: float,
    hidden_dim: int,
) -> float:
    """
    Self-calibrating stress weight for the geometry loss.

    Formula:
        theoretical_stress = 1 / hidden_dim  (optimal decorrelation)
        stress_ratio = current / theoretical
        stress_weight = clamp(ratio / 100, 0.1, 10.0)

    If stress is much higher than theoretical optimum, increase the
    weight to push harder toward decorrelation. If already near optimal,
    use a moderate weight.

    Brain analog: Homeostatic plasticity — the strength of the learning
    signal is proportional to how far the system is from its set point.
    A neuron far from its target firing rate adjusts faster.

    Args:
        current_stress: Current average stress |S_ij|
        hidden_dim: Number of features

    Returns:
        stress_weight: Adaptive weight for the stress/geometry loss
    """
    theoretical = 1.0 / hidden_dim
    if current_stress < 1e-10:
        return 0.1  # Minimum weight

    ratio = current_stress / theoretical
    return max(0.1, min(10.0, ratio / 100.0))
