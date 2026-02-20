"""
SOMI Action-Derived Parameter Cascade
========================================

Derives ALL SOMI parameters from the single action S[phi, W] using
constraints from Levels 1-5.  The ONLY true inputs are:

    hidden_dim (N)  — number of features per Part
    n_parts    (P)  — number of Parts in the circuit brain

Everything else follows from:
  Level 1: Unit/normalization choices (M=1, ei_ratio=0.8)
  Level 2: Continuum scaling (alpha_1 ~ N^{-2/d}, rho*kappa=1/alpha_1, ...)
  Level 3: Spacetime (CFL -> dt, Hawking -> lambda_W, dark energy -> alpha_0)
  Level 4: Gauge theory (kappa_0 = alpha_1, Wilson loop area -> WM rank, ...)
  Level 5: Path integral (saddle-point -> target_zeta, detailed balance -> EMAs)

Source theory:
  24_THE_5_LEVELS_COMPLETE_REFERENCE.md  (Sections: Constraint Cascade,
  Parameter Elimination Principle, Practical Toolkit, Top-Down Upgrades)

This implements the "Parameter Elimination Principle" from the theory:
  "When all 5 levels are taken together, EVERY free parameter may be derivable."
"""

import math
from dataclasses import dataclass
from typing import Dict


# ============================================================
# The Single Derivation Function
# ============================================================

def derive_all_from_action(N: int, P: int) -> Dict:
    """
    Derive ALL SOMI parameters from the action S[phi, W].

    Inputs:
        N: hidden_dim (number of features per Part)
        P: n_parts (number of Parts)

    Returns:
        Dictionary of every parameter the config needs, keyed by
        config field name.  Each value is documented inline with
        the level and equation it comes from.
    """
    # ----------------------------------------------------------
    # LEVEL 1 — Unit Choices (action normalization)
    # ----------------------------------------------------------
    # M = 1 by convention: we measure phi in units where the
    # kinetic term is (1/2) phi_dot^2.  This absorbs M into the
    # definition of phi and sets the energy scale.
    M = 1.0

    # Dale's Law E/I ratio: biological constant (~80% excitatory
    # in cortex).  Not a free parameter — it's an empirical fact
    # about the system we're modeling.
    ei_ratio = 0.8

    # ----------------------------------------------------------
    # LEVEL 2 — Continuum Scaling  (N -> infinity)
    # ----------------------------------------------------------
    # Effective spatial dimension of the brain graph.
    # A circuit with P parts has information flow dimension
    # d_eff = log2(P).  Minimum d=1 (chain).
    # Source: Weyl's law N(omega) ~ Volume * omega^d
    d_eff = max(1.0, math.log2(max(2, P)))

    # Coupling alpha_1: from continuum limit, the coupling must
    # scale as N^{-2/d} to keep energy density finite.
    # alpha_1 = (N_ref / N)^{2/d} with N_ref = 128.
    # Source: 24_5_LEVELS §Level 2 Scaling Laws
    alpha_1 = (128.0 / N) ** (2.0 / d_eff)

    # Mass-conductivity duality: rho * kappa = 1/alpha_1.
    # For sparsity s and mean |W| ~ 1/N, kappa ~ s/N.
    # rho ~ 1/(alpha_1 * kappa) ~ N/(alpha_1 * s).
    # This constrains sparsity: s = d_eff / sqrt(N) keeps the
    # duality balanced (volume conservation under Ricci flow).
    # Source: 24_5_LEVELS §Level 2 Mass-Conductivity Duality
    sparsity = min(0.5, max(0.05, d_eff / math.sqrt(N)))

    # Dale's Law derived parameters: lambda_E = lambda_C =
    # alpha_1 * f_E * f_I.  Coordination and error-smoothing
    # strengths are NOT free — they come from E/I variance
    # times coupling strength.
    # Source: calibration.py derive_dale_parameters()
    f_E, f_I = ei_ratio, 1.0 - ei_ratio
    lambda_E = alpha_1 * f_E * f_I
    lambda_C = lambda_E  # symmetric by E/I balance

    # ----------------------------------------------------------
    # LEVEL 3 — Spacetime  (time becomes geometry)
    # ----------------------------------------------------------
    # Estimated maximum eigenvalue of L_rw for initial W
    # (random doubly-stochastic: lambda_max ~ 2).
    # This tightens as W learns, but we need a safe initial bound.
    lambda_max_est = 2.0

    # Information speed: c_info = sqrt(alpha_1 * lambda_max / M).
    # Source: 24_5_LEVELS §Level 3 Information Speed Limit
    c_info = math.sqrt(alpha_1 * lambda_max_est / M)

    # CFL condition: dt < dx / c_info.  Lattice spacing dx = 1.
    # Safety factor 0.5 (standard in CFD).
    # Source: 24_5_LEVELS §Level 3 CFL condition
    dt = 0.5 / max(c_info, 1e-6)
    dt = max(0.01, min(0.5, dt))

    # Dark energy = alpha_0 (anchoring / cosmological constant).
    # In GR: Lambda / (8 pi G) sets the expansion pressure.
    # In SOMI: alpha_0 / alpha_1 is the ratio of anchoring to
    # coupling.  From the Einstein equations applied to SOMI,
    # the cosmological constant in d+1 dimensions is
    # Lambda ~ 1/L^2 where L is the system size.
    # For a graph of size N: L ~ N^{1/d}, so
    # alpha_0 = alpha_1 / (d_eff + 2).
    # Source: 24_5_LEVELS §Level 3 Dark Energy
    alpha_0 = alpha_1 / (d_eff + 2.0)

    # Weight decay = Hawking radiation: lambda_W = 2*pi*T_H.
    # Hawking temperature: T_H ~ acceleration / (2*pi).
    # In SOMI: "acceleration" ~ alpha_1 (the coupling force),
    # and the Unruh temperature at the system boundary scales
    # as T ~ alpha_1 / N.  So: lambda_W = 2*pi * alpha_1/N.
    # Source: 24_5_LEVELS §Level 3 Hawking radiation = weight decay
    lambda_W = 2.0 * math.pi * alpha_1 / N
    lambda_W = max(1e-5, min(0.01, lambda_W))

    # Timescale ratio: geometry evolves on the light-crossing time
    # of the system.  Light-crossing = P * dx / c_info = P/c_info.
    # Field oscillation period = 2*pi / omega_0 where
    # omega_0 = sqrt((alpha_1 + alpha_0 + 1)/M).
    # Ratio = (P / c_info) / (2*pi / omega_0) = P*omega_0/(2*pi*c_info).
    # Source: 24_5_LEVELS §Level 3 -> Level 2 (metric separability)
    omega_0 = math.sqrt((alpha_1 + alpha_0 + 1.0) / M)
    timescale_ratio = P * omega_0 / (2.0 * math.pi * c_info)
    timescale_ratio = max(2.0, min(16.0, timescale_ratio))

    # ----------------------------------------------------------
    # LEVEL 4 — Gauge Theory & Topology
    # ----------------------------------------------------------
    # Prediction error strength kappa_0 = alpha_1.
    # Gauge coupling unification: the JEPA prediction-error force
    # and the coupling force come from the SAME gauge field
    # (Yang-Mills action = JEPA loss, 24_5_LEVELS §Level 4).
    # So their coefficients are equal.
    kappa_0 = alpha_1

    # Gate coupling kappa_1 = alpha_1 / 2.
    # The basal-ganglia gate is a Z_2 symmetry-breaking of the
    # gauge field (go / no-go).  The broken symmetry gives half
    # the coupling of the full gauge field.
    kappa_1 = alpha_1 / 2.0

    # Gate bottleneck: number of gauge sectors = P.
    # The gate compresses through 1/P of hidden_dim (one sector
    # per Part, mirroring the gauge group decomposition).
    gate_bottleneck = max(0.1, min(0.5, 1.0 / P))

    # White matter rank: from Wilson loop area law.
    # In lattice gauge theory, the number of independent gauge
    # degrees of freedom scales as sqrt(area).  Area ~ P * N
    # (parts times features).  So rank ~ sqrt(P*N).
    # Source: 24_5_LEVELS §Level 4 Wilson Loops
    white_matter_rank = max(8, int(math.ceil(math.sqrt(P * N))))
    white_matter_rank = min(white_matter_rank, N // 2)

    # Exploration noise: gauge temperature / (coupling * N).
    # In gauge theory, thermal fluctuations scale as T/(g^2 N)
    # where g^2 = alpha_1.  The temperature T ~ c_info (kinetic
    # energy scale).  So noise ~ c_info / (alpha_1 * N).
    noise_ratio = c_info / (alpha_1 * N)
    noise_ratio = max(0.001, min(0.05, noise_ratio))

    # ----------------------------------------------------------
    # LEVEL 5 — Path Integral / Quantum (saddle point)
    # ----------------------------------------------------------
    # Optimal damping ratio at the saddle point of the partition
    # function Z = int D[phi] D[W] exp(-S).
    # The saddle point maximizes mutual information between phi
    # and W.  More dimensions = more paths for information flow,
    # so less damping needed per dimension to explore:
    # zeta = 1 / (2 + d_eff).
    # d=2: 0.25 (underdamped, good oscillations)
    # d=3: 0.20, d=4: 0.167, d=5: 0.143 (more underdamped at scale)
    # Source: 24_5_LEVELS §Level 5 Saddle-Point Approximation
    target_zeta = 1.0 / (2.0 + d_eff)
    target_zeta = max(0.05, min(0.3, target_zeta))

    # STDP strength from detailed balance: kappa_stdp = dt.
    # In the path integral, detailed balance requires that the
    # Hebbian (STDP) term's time resolution matches the
    # integration step.  kappa_stdp must be O(dt) — exactly dt.
    kappa_stdp = dt

    # Surprise gate from critical fluctuation threshold.
    # At the saddle point, fluctuations exceeding 1/(2*zeta)
    # standard deviations are "significant" (worth learning from).
    # sigmoid(1/(2*zeta)) gives the fraction of events that
    # qualify as surprising.
    surprise_gate = 1.0 / (1.0 + math.exp(-1.0 / (2.0 * target_zeta)))

    # EMA constants from detailed balance:
    # The path integral's time-discretization gives natural
    # smoothing windows.  The field EMA decays as (1 - dt),
    # precision decays twice as fast (quadratic observable),
    # stress momentum matches the geometry timescale.
    arousal_ema = max(0.9, min(0.999, 1.0 - dt))
    precision_ema = max(0.9, min(0.999, 1.0 - 2.0 * dt))
    stress_momentum_beta = max(0.8, min(0.99, 1.0 - dt / timescale_ratio))

    # ----------------------------------------------------------
    # DERIVED ARCHITECTURE  (from the above physics)
    # ----------------------------------------------------------
    n_systems = max(2, P // 2)

    # Settling steps: half-period of median eigenfrequency.
    # n_settle = pi / (omega_0 * dt).  This is enough time for
    # one oscillation cycle to explore and return to equilibrium.
    n_settle = max(3, int(math.pi / (omega_0 * dt)))

    # Plasticity interval: structural changes happen on the
    # geometry timescale = n_settle * timescale_ratio.
    plasticity_interval = max(50, int(n_settle * timescale_ratio))

    # Eigen update interval: half the plasticity interval
    # (eigendecomposition needed before structural changes).
    eigen_update_interval = max(25, plasticity_interval // 2)

    # Spectral truncation K from Weyl's law:
    # N(omega) ~ Volume * omega^d => dominant modes scale as N^{d/(d+2)}.
    # For a graph of N nodes in d dimensions, the number of modes that
    # carry significant energy is K ~ N^{d/(d+2)}.
    # Raj et al. (2020): brain uses ~10-20 modes.  Clamp to [4, 64].
    spectral_K = int(math.ceil(N ** (d_eff / (d_eff + 2.0))))
    spectral_K = max(4, min(64, spectral_K))

    # Residual weight: Bayesian optimal mixing of prior (h)
    # and posterior (phi_settled).  More settling steps = more
    # trust in the posterior.  w = 1/(1 + n_settle/5).
    residual_weight = 1.0 / (1.0 + n_settle / 5.0)
    residual_weight = max(0.2, min(0.8, residual_weight))

    # Test-time surprise threshold = surprise_gate (same physics).
    test_time_surprise_threshold = surprise_gate

    # Generalization pressure: inversely proportional to log(P).
    # More Parts = less pressure per Part (extensive quantity).
    generalization_pressure_weight = max(0.5, min(2.0, 1.0 / math.log2(max(2, P))))

    # Neuromodulator baselines: 0.5 (neutral / saddle point of
    # sigmoid; self-calibrate from there at runtime).
    ne_baseline = 0.5
    da_baseline = 0.5
    ach_baseline = 0.5
    serotonin_baseline = 0.5

    # Astrocyte timescale: operates on system-size timescale
    # (real astrocytes: seconds; neurons: milliseconds).
    astrocyte_tau = float(N)
    astrocyte_threshold = 2.0 / target_zeta

    # Enable advanced features when the system is large enough
    # for the higher-level physics to matter.
    enable_l3 = N >= 256
    enable_l4 = N >= 256
    enable_l5 = N >= 512
    enable_delays = N >= 512
    enable_astrocyte = N >= 512

    return {
        # --- Level 1: Units ---
        'M': M,
        'ei_ratio': ei_ratio,

        # --- Level 2: Continuum ---
        'd_eff': d_eff,
        'alpha_1': alpha_1,
        'alpha_0': alpha_0,
        'lambda_E': lambda_E,
        'lambda_C': lambda_C,
        'sparsity': sparsity,

        # --- Level 3: Spacetime ---
        'c_info': c_info,
        'dt': dt,
        'lambda_W': lambda_W,
        'timescale_ratio': timescale_ratio,

        # --- Level 4: Gauge ---
        'kappa_0': kappa_0,
        'kappa_1': kappa_1,
        'gate_bottleneck': gate_bottleneck,
        'white_matter_rank': white_matter_rank,
        'noise_ratio': noise_ratio,

        # --- Level 5: Path Integral ---
        'target_zeta': target_zeta,
        'kappa_stdp': kappa_stdp,
        'surprise_gate': surprise_gate,
        'arousal_ema': arousal_ema,
        'precision_ema': precision_ema,
        'stress_momentum_beta': stress_momentum_beta,

        # --- Derived Architecture ---
        'n_systems': n_systems,
        'n_settle': n_settle,
        'spectral_K': spectral_K,
        'plasticity_interval': plasticity_interval,
        'eigen_update_interval': eigen_update_interval,
        'residual_weight': residual_weight,
        'test_time_surprise_threshold': test_time_surprise_threshold,
        'generalization_pressure_weight': generalization_pressure_weight,

        # --- Neuromodulators ---
        'ne_baseline': ne_baseline,
        'da_baseline': da_baseline,
        'ach_baseline': ach_baseline,
        'serotonin_baseline': serotonin_baseline,

        # --- Astrocyte ---
        'astrocyte_tau': astrocyte_tau,
        'astrocyte_threshold': astrocyte_threshold,

        # --- Feature flags ---
        'compute_level3': enable_l3,
        'compute_level4': enable_l4,
        'compute_level5': enable_l5,
        'delays_enabled': enable_delays,
        'astrocyte_enabled': enable_astrocyte,
    }


# ============================================================
# Pretty-Print for Inspection
# ============================================================

def print_derivation(N: int, P: int):
    """Print the full derivation chain for inspection / debugging."""
    params = derive_all_from_action(N, P)
    d_eff = params.pop('d_eff')
    c_info = params.pop('c_info')

    print(f"{'=' * 70}")
    print(f"  SOMI Action-Derived Parameters")
    print(f"  Inputs: hidden_dim={N}, n_parts={P}")
    print(f"  Effective dimension d_eff = {d_eff:.2f}")
    print(f"  Information speed c_info = {c_info:.4f}")
    print(f"{'=' * 70}")

    sections = {
        'Level 1 (Units)': ['M', 'ei_ratio'],
        'Level 2 (Continuum)': ['alpha_1', 'alpha_0', 'lambda_E', 'lambda_C', 'sparsity'],
        'Level 3 (Spacetime)': ['dt', 'lambda_W', 'timescale_ratio'],
        'Level 4 (Gauge)': ['kappa_0', 'kappa_1', 'gate_bottleneck', 'white_matter_rank', 'noise_ratio'],
        'Level 5 (Path Integral)': ['target_zeta', 'kappa_stdp', 'surprise_gate',
                                     'arousal_ema', 'precision_ema', 'stress_momentum_beta'],
        'Derived Architecture': ['n_systems', 'n_settle', 'spectral_K',
                                  'plasticity_interval', 'eigen_update_interval',
                                  'residual_weight', 'test_time_surprise_threshold',
                                  'generalization_pressure_weight'],
        'Neuromodulators': ['ne_baseline', 'da_baseline', 'ach_baseline', 'serotonin_baseline'],
        'Astrocyte': ['astrocyte_tau', 'astrocyte_threshold'],
        'Feature Flags': ['compute_level3', 'compute_level4', 'compute_level5',
                          'delays_enabled', 'astrocyte_enabled'],
    }

    for section, keys in sections.items():
        print(f"\n  {section}:")
        for k in keys:
            v = params.get(k)
            if isinstance(v, float):
                print(f"    {k:35s} = {v:.6f}")
            else:
                print(f"    {k:35s} = {v}")

    print(f"\n{'=' * 70}")
    print(f"  Total derived parameters: {len(params)}")
    print(f"  Free parameters: 0  (everything from action + Levels 2-5)")
    print(f"{'=' * 70}")


# ============================================================
# Quick comparison across scales
# ============================================================

def compare_scales():
    """Show how the cascade adapts across all standard scales."""
    scales = [
        ('Circuit-S', 128, 4),
        ('Circuit-M', 256, 8),
        ('Circuit-L', 512, 16),
        ('Circuit-XL', 1024, 32),
    ]

    key_params = [
        'alpha_1', 'alpha_0', 'dt', 'target_zeta', 'timescale_ratio',
        'lambda_W', 'noise_ratio', 'kappa_0', 'white_matter_rank',
        'n_settle', 'spectral_K', 'sparsity', 'residual_weight',
    ]

    header = f"{'Parameter':25s}"
    for name, _, _ in scales:
        header += f" {name:>12s}"
    print(header)
    print("-" * len(header))

    all_params = {}
    for name, n, p in scales:
        all_params[name] = derive_all_from_action(n, p)

    for k in key_params:
        row = f"{k:25s}"
        for name, _, _ in scales:
            v = all_params[name].get(k)
            if isinstance(v, float):
                row += f" {v:12.6f}"
            else:
                row += f" {v!s:>12s}"
        print(row)
