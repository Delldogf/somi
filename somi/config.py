"""
SOMI 4.0 Brain Configuration
=============================

All parameters are REGIME CHOICES (physical constants), not hyperparameters.
They define what KIND of system SOMI is, not how well it performs.

Everything that CAN be self-calibrated IS self-calibrated:
  beta, n_settle, eta, arousal, stress_weight, mass, lambda_E/C

Based on: somi_2_0/config.py (expanded for circuit brain architecture)
Source theory: SOMI_3_0/theory/24_THE_5_LEVELS_COMPLETE_REFERENCE.md

Scale presets:
  Circuit-S:  4 Parts, 2 Systems, hidden=128   (~2M params)   - dev/testing
  Circuit-M:  8 Parts, 4 Systems, hidden=256   (~15M params)  - small experiments
  Circuit-L:  16 Parts, 8 Systems, hidden=512  (~100M params) - real tasks
  Circuit-XL: 32 Parts, 16 Systems, hidden=1024 (~400M params) - production
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


@dataclass
class SOMIBrainConfig:
    """
    Complete configuration for a SOMI 4.0 Circuit Brain.

    Parameters are organized by:
    1. Architecture (dimensions, circuit structure)
    2. Field physics (forces, integration)
    3. Geometry learning (stress, plasticity)
    4. Self-calibration (auto-tuned from physics)
    5. Neuromodulators (4 brain-inspired systems)
    6. Paper-sourced enhancements (from 80+ papers)
    7. Diagnostics control (which diagnostics to compute)
    8. Training mode (local-only vs hybrid)
    9. Cost/safety (prevent expensive accidents)
    """

    # =========================================================================
    # 1. ARCHITECTURE
    # =========================================================================

    hidden_dim: int = 128
    """Dimension of activity field phi in each Part.
    Each Part has its own phi of this size. Like the number of neurons
    in a brain region."""

    n_parts: int = 4
    """Number of Parts (brain regions) in the circuit.
    Each Part has its own W_local, phi, phi_dot, mass, neuromodulators."""

    n_systems: int = 2
    """Number of Systems (circuits) that route through Parts.
    Systems SHARE Parts — this creates generalization pressure.
    NOT Mixture of Experts: MoE routes to separate experts.
    SOMI circuits share Parts, forcing them to generalize."""

    system_routes: Optional[List[List[int]]] = None
    """Which Parts each System routes through. None = auto-generate.
    Example: [[0, 1, 2], [1, 2, 3]] means System 0 uses Parts 0,1,2
    and System 1 uses Parts 1,2,3 (Parts 1,2 are SHARED).
    Shared Parts receive stress from ALL Systems = generalization pressure."""

    shared_part_ids: Optional[List[int]] = None
    """Which Parts are shared across multiple Systems. None = auto-detect.
    Shared Parts are like PFC in the brain — they appear in every circuit
    and must learn representations that work for all tasks."""

    white_matter_rank: int = 32
    """Rank of low-rank White Matter connections between Parts.
    White Matter = gauge connections (Level 4 math).
    Lower rank = fewer parameters, more constrained."""

    # =========================================================================
    # 2. FIELD PHYSICS (Forces + Integration)
    # =========================================================================

    # --- Mass and Damping ---

    M: float = 1.0
    """Base inertial mass. Brain analog: membrane capacitance.
    Higher M = slower response. Per-node mass M_i is computed from
    eigenvalues (M_i = 1/omega_i) and Herfindahl index."""

    target_zeta: float = 0.15
    """Target damping ratio (dimensionless). Brain analog: membrane time constant.
    0.15 = underdamped = oscillations persist for several cycles.
    Beta is SELF-CALIBRATED from this: beta_i = 2 * zeta * sqrt(M_i * K_ii)."""

    # --- Coupling (9 Field Forces) ---

    alpha_1: float = 1.0
    """Coupling strength through W (Laplacian force).
    SCALES with model size: alpha_1 ~ N^(-2/d) from Level 2.
    Brain analog: synaptic conductance."""

    alpha_0: float = 0.1
    """Local anchoring strength (pull toward zero).
    Level 3 interpretation: cosmological constant (dark energy).
    Brain analog: leak conductance."""

    kappa_0: float = 1.0
    """Precision-weighted prediction error strength (JEPA force).
    Scales the phi - phi_hat force."""

    kappa_1: float = 0.5
    """Basal ganglia gate strength.
    Controls the bottleneck gate on total force."""

    gate_bottleneck: float = 0.25
    """Gate compression ratio for basal ganglia.
    Fraction of hidden_dim used in the gate's bottleneck layer."""

    noise_ratio: float = 0.003
    """Exploration noise (relative to signal). Brain analog: spontaneous activity.
    Allows escape from local minima during settling."""

    # --- Integration ---

    dt: float = 0.15
    """Time step for settling dynamics.
    CFL condition (Level 3): dt < dx / c_info_max. When use_cfl=True,
    dt is automatically clamped to satisfy this."""

    n_settle: int = -1
    """Settling steps per forward pass. -1 = auto-compute from eigenfrequencies.
    Auto formula: n_settle = max(3, floor(pi / (omega_median * dt))).
    Typical optimal: 5-10 steps."""

    residual_weight: float = 0.5
    """Output mixing: output = residual_weight * h + (1-residual_weight) * phi_settled."""

    # =========================================================================
    # 3. GEOMETRY LEARNING (Stress Tensor + Plasticity)
    # =========================================================================

    # --- Geometry Equation: W_dot = -eta * S - lambda_W * W + kappa_stdp * phi * phi_dot ---

    timescale_ratio: float = 8.0
    """Geometry learns 8x slower than field dynamics.
    CRITICAL: Was 2.0, increased to 8.0 after SOMI 2.0 experiments.
    Too fast (2.0) = network fragmentation. Too slow (16.0) = no learning.
    Eta is computed as: eta = (0.1 / timescale_ratio) * (0.5 + arousal)."""

    lambda_W: float = 0.001
    """Weight decay (metabolic cost). Brain analog: synaptic pruning during sleep.
    Level 3 interpretation: Hawking radiation (lambda_W = 2*pi*T_H)."""

    lambda_E: float = 0.1
    """Error mismatch weight in stress tensor.
    S_ij = lambda_E * E_ij - lambda_C * C_ij + kinetic + STDP.
    When dales_law=True: auto-computed from alpha_1 and ei_ratio."""

    lambda_C: float = 0.1
    """Coordination weight in stress tensor.
    S_ij = lambda_E * E_ij - lambda_C * C_ij + kinetic + STDP.
    When dales_law=True: auto-computed from alpha_1 and ei_ratio."""

    kappa_stdp: float = 0.15
    """STDP temporal correlation strength. Must be O(dt).
    T_ij = phi_i * phi_dot_j (ASYMMETRIC — do NOT symmetrize).
    Brain analog: spike-timing-dependent plasticity."""

    # --- Dale's Law (E/I Balance) ---

    dales_law: bool = True
    """Enable Dale's Law: W = W_excitatory - W_inhibitory, both non-negative.
    Enforced via Signed Sinkhorn normalization.
    When True: lambda_E, lambda_C auto-computed from alpha_1 and ei_ratio."""

    ei_ratio: float = 0.8
    """Fraction of excitatory features (80% E, 20% I).
    Brain: ~80% glutamate (excitatory), ~20% GABA (inhibitory)."""

    # --- Structural Plasticity ---

    sparsity: float = 0.1
    """Target connectivity fraction (~10% like cortex)."""

    plasticity_interval: int = 100
    """Steps between prune/grow events. Every 100 steps:
    prune lowest 5% of W, grow new edges where stress is highest."""

    # =========================================================================
    # 4. SELF-CALIBRATION (Auto-tuned from physics)
    # =========================================================================

    arousal_ema: float = 0.99
    """EMA decay for arousal running average.
    Arousal = sigmoid(error_mag / running_avg - 1)."""

    precision_ema: float = 0.95
    """EMA decay for precision (error variance) tracking.
    Pi = diag(1/var(e)) for precision-weighted error forces."""

    stress_momentum_beta: float = 0.9
    """EMA for stress tensor (Titans paper: momentum on geometry updates).
    S_t = stress_momentum_beta * S_{t-1} + (1-stress_momentum_beta) * S_current.
    Smooths noisy stress for more stable geometry learning."""

    eigen_update_interval: int = 50
    """Steps between eigendecomposition recomputation.
    Only recompute when W changes significantly (saves compute)."""

    spectral_K: int = 0
    """Number of eigenmodes to keep for spectral/SSM settling.
    0 = use all modes (no truncation).
    Derived from Weyl's law: K ~ N^{d/(d+2)}.
    Raj et al. (2020): brain uses ~10-20 modes."""

    use_spectral_settling: bool = True
    """Auto-select SSM settling when eigenmodes are available.
    Falls back to symplectic if eigendecomposition fails."""

    # =========================================================================
    # 5. NEUROMODULATORS (4 brain-inspired monitoring systems)
    # =========================================================================

    neuromodulators_enabled: bool = True
    """Enable the 4 neuromodulator systems.
    NE (norepinephrine): arousal -> beta (damping)
    DA (dopamine): reward -> eta (learning rate)
    ACh (acetylcholine): attention -> mass (processing speed)
    5-HT (serotonin): mood -> n_settle (processing depth)"""

    ne_baseline: float = 0.5
    """Norepinephrine baseline. Range [0, 1]."""

    da_baseline: float = 0.5
    """Dopamine baseline. Range [0, 1]."""

    ach_baseline: float = 0.5
    """Acetylcholine baseline. Range [0, 1]."""

    serotonin_baseline: float = 0.5
    """Serotonin baseline. Range [0, 1]."""

    # =========================================================================
    # 6. PAPER-SOURCED ENHANCEMENTS
    # =========================================================================

    # --- Conduction Delays (HORN / Raj papers) ---

    delays_enabled: bool = False
    """Enable per-edge conduction delays tau_ij.
    HORN (PNAS 2025): heterogeneous delays create temporal structure.
    Raj et al. 2020: frequency-dependent phase from delays.
    tau_ij = 1/|W_ij| (strong connections = fast, weak = slow)."""

    delay_range: Tuple[float, float] = (0.0, 1.0)
    """Min and max delay values when delays_enabled=True."""

    # --- Astrocyte Modulation (Freeman, Ahrens, Papouin 2025) ---

    astrocyte_enabled: bool = False
    """Enable astrocyte-like supervisory layer.
    Accumulates surprise over slow timescale (calcium-like).
    When threshold crossed: modulates alpha_1, beta, eta."""

    astrocyte_tau: float = 100.0
    """Astrocyte time constant (slow, like real astrocytes: seconds to minutes).
    Controls how quickly the calcium accumulator responds."""

    astrocyte_threshold: float = 2.0
    """Surprise threshold for astrocyte modulation activation."""

    # --- Titans Momentum (Behrouz 2024) ---

    data_dependent_forgetting: bool = False
    """Enable data-dependent forgetting from Titans paper.
    alpha_t ~ 0 for same context, ~1 for context change.
    Geometry forgets in new contexts, retains in stable ones."""

    # --- CTM Temporal Decay (Darlow 2025) ---

    ctm_temporal_decay: bool = False
    """Enable learnable per-edge temporal decay for coordination C_ij.
    From Continuous Thought Machines paper.
    Allows different edges to have different memory timescales."""

    # --- LinOSS Parallel Scan (Rusch & Rus, ICLR 2025) ---

    use_parallel_scan: bool = False
    """Use associative parallel scan for O(log n) settling.
    Alternative to sequential symplectic Euler.
    Implicit-explicit discretization: A (coupling) implicit, B (input) explicit."""

    # --- Vanchurin Dynamic Alpha (Geometric Learning Regimes) ---

    dynamic_alpha_enabled: bool = False
    """Enable 3-phase learning regime transitions.
    Phase 1 (exploration): alpha = 1.0 (maximum exploration)
    Phase 2 (efficient): alpha = 0.5 (balanced)
    Phase 3 (consolidation): alpha = 0.0 (lock in knowledge)"""

    alpha_schedule: Tuple[float, float, float] = (1.0, 0.5, 0.0)
    """Alpha values for the 3 learning phases."""

    # --- Entropy Constraint (Vanchurin Second Law of Learning) ---

    enforce_entropy_decrease: bool = True
    """Enforce dS/dt <= 0 during learning (entropy never increases).
    From Vanchurin 2020: 'The World as a Neural Network.'"""

    # --- Test-Time Learning (Titans) ---

    test_time_learning: bool = True
    """Allow W_local updates during inference via SOMI stress.
    This is what makes SOMI fundamentally different from transformers:
    it learns at test time without backprop."""

    test_time_surprise_threshold: float = 0.5
    """Only update W at test time if surprise exceeds this threshold.
    Prevents unnecessary updates during confident predictions."""

    surprise_gate: float = 0.0
    """Minimum arousal to allow geometry updates during training.
    From SOMIAdaptive: only update W when the model is actually surprised.
    0.0 = always update (default for backward compat).
    0.3 = only update when arousal > 0.3 (recommended for auto mode).
    Brain analog: dopamine gates when to learn."""

    # --- Generalization Pressure ---

    generalization_pressure_weight: float = 1.0
    """Weight for stress from multiple Systems on shared Parts.
    Higher = stronger pressure for shared Parts to generalize."""

    _action_derived: bool = False
    """Internal flag: True when auto() used the action cascade.
    Prevents __post_init__ from re-scaling already-derived params."""

    # =========================================================================
    # 7. DIAGNOSTICS CONTROL
    # =========================================================================

    compute_level1: bool = True
    """Compute Level 1 (standard) diagnostics: stress, mass, Hamiltonian, etc."""

    compute_level2: bool = True
    """Compute Level 2 (continuum) diagnostics: mass-conductivity, spectral, etc."""

    compute_level3: bool = False
    """Compute Level 3 (spacetime) diagnostics: CFL, c_info, geodesics.
    More expensive — disable for fast iteration."""

    compute_level4: bool = False
    """Compute Level 4 (gauge) diagnostics: Wilson loops, curvature, Chern-Simons.
    Most expensive — enable for deep analysis only."""

    compute_level5: bool = False
    """Compute Level 5 (quantum) diagnostics: ensemble, saddle-point.
    Research-level — disable for normal use."""

    compute_neuroscience: bool = True
    """Compute neuroscience diagnostics (10 brain-inspired tests)."""

    compute_circuit: bool = True
    """Compute circuit diagnostics (11 metrics including shared-Part)."""

    use_cfl_condition: bool = True
    """Enforce CFL stability condition on dt (Level 3).
    dt_safe = dx / c_info_max. dt is clamped when this is True."""

    # =========================================================================
    # 8. TRAINING MODE
    # =========================================================================

    pure_local: bool = False
    """Phase 2 mode: Only local learning (no backprop at all).
    False = Phase 3 hybrid (macro backprop + micro SOMI).
    True = Phase 2 local-only (all learning via stress/STDP)."""

    use_bfloat16: bool = True
    """Use bfloat16 for mixed precision (NOT float16 — eigendecomp needs range).
    float16 causes NaN in eigenvalue computation."""

    # =========================================================================
    # 9. COST / SAFETY
    # =========================================================================

    max_steps: int = 5000
    """Maximum training steps before cost lock triggers.
    Set SOMI_EXPENSIVE_RUN=true to override."""

    max_hidden_dim: int = 512
    """Maximum hidden_dim before cost lock triggers.
    Set SOMI_EXPENSIVE_RUN=true to override."""

    # =========================================================================
    # AUTO-SCALING (computed from other parameters)
    # =========================================================================

    def __post_init__(self):
        """Auto-compute derived parameters after initialization."""
        if not self._action_derived:
            # Level 2 scaling: alpha_1 ~ N^(-2/d) where d=1 for 1D
            if self.hidden_dim != 128:
                scale = 128.0 / self.hidden_dim
                self.alpha_1 = self.alpha_1 * scale

            # Dale's Law auto-computation
            if self.dales_law:
                dale_param = self.alpha_1 * self.ei_ratio * (1.0 - self.ei_ratio)
                self.lambda_E = dale_param
                self.lambda_C = dale_param

        # Auto-generate system routes if not specified
        if self.system_routes is None:
            self.system_routes = self._auto_system_routes()

        # Auto-detect shared parts
        if self.shared_part_ids is None:
            self.shared_part_ids = self._auto_detect_shared_parts()

    def _auto_system_routes(self) -> List[List[int]]:
        """Generate default system routes with overlapping Parts.
        Creates n_systems routes through n_parts, with ~50% overlap.
        This ensures some Parts are shared (generalization pressure)."""
        routes = []
        parts_per_system = max(2, self.n_parts // self.n_systems + 1)
        for s in range(self.n_systems):
            start = (s * (self.n_parts - 1)) // max(1, self.n_systems - 1)
            route = list(range(start, min(start + parts_per_system, self.n_parts)))
            # Always include Part 0 as the "PFC" (shared across all systems)
            if 0 not in route:
                route = [0] + route
            routes.append(route)
        return routes

    def _auto_detect_shared_parts(self) -> List[int]:
        """Find Parts that appear in more than one System."""
        from collections import Counter
        part_counts = Counter()
        for route in self.system_routes:
            for part_id in route:
                part_counts[part_id] += 1
        return [p for p, count in part_counts.items() if count > 1]

    # =========================================================================
    # SCALE PRESETS
    # =========================================================================

    @staticmethod
    def circuit_s() -> 'SOMIBrainConfig':
        """Circuit-S: 4 Parts, 2 Systems, hidden=128.
        For development and testing. ~2M parameters."""
        return SOMIBrainConfig(
            hidden_dim=128,
            n_parts=4,
            n_systems=2,
            white_matter_rank=16,
        )

    @staticmethod
    def circuit_m() -> 'SOMIBrainConfig':
        """Circuit-M: 8 Parts, 4 Systems, hidden=256.
        For small experiments. ~15M parameters."""
        return SOMIBrainConfig(
            hidden_dim=256,
            n_parts=8,
            n_systems=4,
            white_matter_rank=32,
        )

    @staticmethod
    def circuit_l() -> 'SOMIBrainConfig':
        """Circuit-L: 16 Parts, 8 Systems, hidden=512.
        For real tasks. ~100M parameters."""
        return SOMIBrainConfig(
            hidden_dim=512,
            n_parts=16,
            n_systems=8,
            white_matter_rank=64,
            compute_level3=True,   # enable spacetime diagnostics
            compute_level4=True,   # enable gauge diagnostics
        )

    @staticmethod
    def circuit_xl() -> 'SOMIBrainConfig':
        """Circuit-XL: 32 Parts, 16 Systems, hidden=1024.
        For production. ~400M parameters."""
        return SOMIBrainConfig(
            hidden_dim=1024,
            n_parts=32,
            n_systems=16,
            white_matter_rank=128,
            compute_level3=True,
            compute_level4=True,
            compute_level5=True,   # enable quantum diagnostics
            delays_enabled=True,   # enable conduction delays
            astrocyte_enabled=True,  # enable astrocyte modulation
        )

    @staticmethod
    def auto(hidden_dim: int = 128, n_parts: int = 4) -> 'SOMIBrainConfig':
        """Zero-hyperparameter config: derive EVERYTHING from hidden_dim and n_parts.

        Uses the full Level 1-5 constraint cascade (action_derived.py) to
        compute every parameter from the single SOMI action S[phi, W].

        The ONLY inputs are hidden_dim and n_parts.  Everything else —
        coupling, damping, dt, weight decay, noise, gate, timescales,
        EMAs, sparsity, white-matter rank, feature flags — is derived
        from physics across all 5 levels of the theory.

        Level 1: M=1 (units), ei_ratio=0.8 (Dale's Law)
        Level 2: alpha_1 ~ N^{-2/d}, alpha_0, sparsity, lambda_E/C
        Level 3: dt from CFL, lambda_W from Hawking, timescale_ratio
        Level 4: kappa_0 = alpha_1 (gauge unification), WM rank, noise
        Level 5: target_zeta from saddle point, kappa_stdp = dt, EMAs

        See somi/physics/action_derived.py for full derivation chain.
        """
        from .physics.action_derived import derive_all_from_action

        d = derive_all_from_action(hidden_dim, n_parts)

        return SOMIBrainConfig(
            hidden_dim=hidden_dim,
            n_parts=n_parts,
            n_systems=d['n_systems'],
            white_matter_rank=d['white_matter_rank'],

            # Level 1 (units)
            M=d['M'],
            ei_ratio=d['ei_ratio'],

            # Level 2 (continuum)
            alpha_1=d['alpha_1'],
            alpha_0=d['alpha_0'],
            lambda_E=d['lambda_E'],
            lambda_C=d['lambda_C'],
            sparsity=d['sparsity'],

            # Level 3 (spacetime)
            dt=d['dt'],
            lambda_W=d['lambda_W'],
            timescale_ratio=d['timescale_ratio'],

            # Level 4 (gauge)
            kappa_0=d['kappa_0'],
            kappa_1=d['kappa_1'],
            gate_bottleneck=d['gate_bottleneck'],
            noise_ratio=d['noise_ratio'],

            # Level 5 (path integral)
            target_zeta=d['target_zeta'],
            kappa_stdp=d['kappa_stdp'],
            surprise_gate=d['surprise_gate'],
            arousal_ema=d['arousal_ema'],
            precision_ema=d['precision_ema'],
            stress_momentum_beta=d['stress_momentum_beta'],

            # Derived architecture
            n_settle=d['n_settle'],
            spectral_K=d['spectral_K'],
            use_spectral_settling=True,
            residual_weight=d['residual_weight'],
            plasticity_interval=d['plasticity_interval'],
            eigen_update_interval=d['eigen_update_interval'],
            test_time_surprise_threshold=d['test_time_surprise_threshold'],
            generalization_pressure_weight=d['generalization_pressure_weight'],

            # Neuromodulators
            ne_baseline=d['ne_baseline'],
            da_baseline=d['da_baseline'],
            ach_baseline=d['ach_baseline'],
            serotonin_baseline=d['serotonin_baseline'],

            # Astrocyte
            astrocyte_tau=d['astrocyte_tau'],
            astrocyte_threshold=d['astrocyte_threshold'],

            # Feature flags (from system size)
            compute_level3=d['compute_level3'],
            compute_level4=d['compute_level4'],
            compute_level5=d['compute_level5'],
            delays_enabled=d['delays_enabled'],
            astrocyte_enabled=d['astrocyte_enabled'],

            # Always on
            test_time_learning=True,
            dales_law=True,

            # Mark as action-derived so __post_init__ won't re-scale
            _action_derived=True,
        )

    @staticmethod
    def from_scratch() -> 'SOMIBrainConfig':
        """True zero-input config: start from the smallest viable brain.

        The theory says even hidden_dim and n_parts should be automatic.
        This starts at hidden_dim=16, n_parts=1 (one tiny region) and
        relies on AutoGrowth to grow neurons (hidden_dim) and add Parts
        (n_parts) as stress demands.

        Use this when you want the model to discover its own size.
        Pair with AutoGrowth (somi.lm.growth) to let it evolve.

        The action cascade still derives every physics parameter from
        hidden_dim=16 and n_parts=1; after each growth event, the
        cascade re-derives parameters for the new size.
        """
        return SOMIBrainConfig.auto(hidden_dim=16, n_parts=1)
