"""
SOMI 4.0 Part — A Single Brain Region
========================================

A Part is like one region of the brain (e.g., visual cortex, motor cortex).
Each Part has:
  - phi: activity field (what the neurons are doing right now)
  - phi_dot: velocity (how fast they're changing)
  - W_local: internal connectivity (how neurons in this region connect)
  - mass: per-feature inertial mass (heavy = slow, light = fast)
  - neuromodulators: NE, DA, ACh, 5-HT levels
  - stress momentum buffer (Titans-style smoothed stress)

Key features from papers:
  - DHO (Damped Harmonic Oscillator) from HORN (PNAS 2025): explicit phi, phi_dot
  - Per-edge conduction delays from HORN/Raj: tau_ij = 1/|W_ij|
  - Astrocyte calcium accumulator from Freeman/Ahrens 2025
  - CTM learnable temporal decay from Darlow 2025
  - Self-calibration of beta, n_settle, eta from physics

A Part can be shared across multiple Systems (brain circuits).
When shared, it receives stress from ALL systems, creating "generalization
pressure" — it must learn representations that work for ALL tasks.

Source: SOMI_3_0/theory/02_THE_CIRCUIT_BRAIN.md
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import math

from ..config import SOMIBrainConfig
from ..physics.forces import compute_laplacian, PrecisionTracker, BasalGangliaGate
from ..physics.geometry import (
    compute_stress_tensor,
    compute_kinetic_stress,
    geometry_step,
    initialize_W,
    structural_plasticity,
    StressMomentum,
    mass_conductivity_constraint,
    compute_data_dependent_alpha,
)
from ..physics.hamiltonian import HamiltonianTracker, EntropyTracker
from ..physics.settling import (
    settle,
    compute_eigendecomposition,
    compute_auto_n_settle,
)


class SOMIPart(nn.Module):
    """
    A single brain region (Part) in the SOMI Circuit Brain.

    Each Part is a self-contained physics simulation:
    - It has its own W_local (internal connectivity)
    - It settles its own phi under its own forces
    - It learns its own geometry via stress/STDP
    - It tracks its own Hamiltonian, entropy, and health

    Parts communicate through White Matter (gauge connections between Parts).
    Parts can be SHARED across multiple Systems — this is the key
    architectural innovation of SOMI 4.0.

    Think of a Part like a department in a company:
    - It has its own team (neurons = features in phi)
    - Its own internal processes (W_local = who talks to whom)
    - Its own productivity metrics (stress, Hamiltonian)
    - It can work on multiple projects simultaneously (shared across Systems)

    Args:
        part_id: Unique identifier for this Part
        config: SOMIBrainConfig
        device: Torch device
    """

    def __init__(
        self,
        part_id: int,
        config: SOMIBrainConfig,
        device: torch.device = None,
    ):
        super().__init__()
        self.part_id = part_id
        self.config = config
        self.device = device or torch.device('cpu')
        H = config.hidden_dim

        # ===== Core State =====
        # phi and phi_dot are NOT nn.Parameters — they're dynamic state
        # that changes every forward pass. W_local IS a buffer (not Parameter)
        # because it's updated by local physics rules, not backprop.

        # Initialize W_local and sparsity mask
        W_init, mask_init = initialize_W(H, config.sparsity, self.device)
        self.register_buffer('W_local', W_init)      # [H, H] connectivity
        self.register_buffer('mask', mask_init)        # [H, H] sparsity mask

        # Per-feature mass: initialized from Herfindahl index of W
        mass_init = self._compute_initial_mass(W_init)
        self.register_buffer('mass', mass_init)        # [H] per-feature mass

        # Dale's Law E/I sign mask
        if config.dales_law:
            n_exc = int(H * config.ei_ratio)
            ei = torch.ones(H, device=self.device)
            ei[n_exc:] = -1.0
            self.register_buffer('ei_mask', ei)
        else:
            self.ei_mask = None

        # ===== Components =====
        self.precision_tracker = PrecisionTracker(H, config.precision_ema)
        self.precision_tracker = self.precision_tracker.to(self.device)

        self.gate = BasalGangliaGate(H, config.gate_bottleneck)
        self.gate = self.gate.to(self.device)

        self.stress_momentum = StressMomentum(H, config.stress_momentum_beta, self.device)

        # ===== Trackers =====
        self.hamiltonian_tracker = HamiltonianTracker()
        self.entropy_tracker = EntropyTracker()

        # ===== Eigendecomposition Cache =====
        self.register_buffer('eigenvalues', torch.zeros(H))
        self.register_buffer('eigenvectors', torch.eye(H, device=self.device))
        self.register_buffer('eigen_step', torch.tensor(0))
        self._update_eigen()

        # ===== Self-Calibration State =====
        self.register_buffer('arousal', torch.tensor(0.5))
        self.register_buffer('error_running_avg', torch.tensor(1.0))
        self.register_buffer('global_step', torch.tensor(0))

        # ===== Neuromodulator Levels =====
        if config.neuromodulators_enabled:
            self.register_buffer('ne_level', torch.tensor(config.ne_baseline))
            self.register_buffer('da_level', torch.tensor(config.da_baseline))
            self.register_buffer('ach_level', torch.tensor(config.ach_baseline))
            self.register_buffer('serotonin_level', torch.tensor(config.serotonin_baseline))

        # ===== Astrocyte (optional) =====
        if config.astrocyte_enabled:
            self.register_buffer('calcium', torch.tensor(0.0))

        # ===== Conduction Delays (optional) =====
        if config.delays_enabled:
            # tau_ij = 1/|W_ij| — strong connections are fast
            delays = 1.0 / (W_init.abs().clamp(min=1e-4))
            delays = delays.clamp(*config.delay_range)
            self.register_buffer('delays', delays)

        # Previous stress for data-dependent forgetting
        self._prev_stress = None

    def _compute_initial_mass(self, W: torch.Tensor) -> torch.Tensor:
        """Compute per-feature mass from Herfindahl index of W.

        M_i = M_0 * h_mean / h_i where h_i = sum_j(W_ij^2)
        Features with concentrated connections (high h) are light (fast).
        Features with spread-out connections (low h) are heavy (slow).
        """
        h = (W ** 2).sum(dim=1).clamp(min=1e-8)
        h_mean = h.mean()
        mass = self.config.M * h_mean / h
        return mass.clamp(min=0.1, max=10.0)

    def _update_eigen(self):
        """Update eigendecomposition of L_rw (cached for efficiency).
        Uses spectral_K from the action cascade (Weyl's law) for truncation."""
        L_rw = compute_laplacian(self.W_local)
        K = self.config.spectral_K
        truncation = K if K > 0 and K < self.config.hidden_dim else None
        eigenvalues, eigenvectors, _ = compute_eigendecomposition(L_rw, truncation)
        eigenvalues = eigenvalues.to(self.device)
        eigenvectors = eigenvectors.to(self.device)
        n = min(eigenvalues.shape[0], self.eigenvalues.shape[0])
        self.eigenvalues.zero_()
        self.eigenvalues[:n] = eigenvalues[:n]
        if eigenvectors.shape[1] <= self.eigenvectors.shape[1]:
            self.eigenvectors.zero_()
            self.eigenvectors[:, :eigenvectors.shape[1]] = eigenvectors
        else:
            self.eigenvectors.copy_(eigenvectors[:, :self.eigenvectors.shape[1]])

    def _calibrate(self, error_mag: float):
        """Auto-calibrate beta, eta, n_settle, stress_weight from physics state."""
        # Arousal = sigmoid(error / running_avg - 1)
        self.error_running_avg = (
            self.config.arousal_ema * self.error_running_avg
            + (1 - self.config.arousal_ema) * error_mag
        )
        arousal_input = error_mag / (self.error_running_avg.item() + 1e-8) - 1.0
        self.arousal = torch.sigmoid(torch.tensor(arousal_input, device=self.device))

    @property
    def beta(self) -> torch.Tensor:
        """Self-calibrated damping coefficient per feature.
        beta_i = 2 * zeta * sqrt(M_i * K_ii)
        """
        # K_ii ≈ alpha_1 * L_rw_diag + alpha_0 + 1.0
        L_rw = compute_laplacian(self.W_local)
        K_diag = self.config.alpha_1 * L_rw.diag() + self.config.alpha_0 + 1.0
        beta = 2.0 * self.config.target_zeta * torch.sqrt(
            (self.mass * K_diag.clamp(min=1e-8)).clamp(min=1e-8)
        )
        return beta

    @property
    def eta(self) -> float:
        """Self-calibrated geometry learning rate.
        eta = (0.1 / timescale_ratio) * (0.5 + arousal)
        """
        return (0.1 / self.config.timescale_ratio) * (0.5 + self.arousal.item())

    @property
    def n_settle(self) -> int:
        """Self-calibrated settling steps from eigenfrequencies."""
        if self.config.n_settle > 0:
            return self.config.n_settle
        return compute_auto_n_settle(
            self.eigenvalues, self.config.dt, self.config.target_zeta
        )

    def forward(
        self,
        h: torch.Tensor,
        phi_target: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Run one settling episode in this Part.

        Steps:
        1. phi starts at h (input from X-encoder or White Matter)
        2. Compute forces and settle phi (symplectic Euler or spectral)
        3. Compute stress tensor S
        4. Update geometry W_local (local learning — no backprop!)
        5. Return settled phi and diagnostics

        Args:
            h: [batch, seq, hidden] input hidden state
            phi_target: [batch, seq, hidden] target (None = use h itself)
            training: Whether in training mode

        Returns:
            phi: [batch, seq, hidden] settled field
            phi_dot: [batch, seq, hidden] velocity
            diagnostics: Dict with all metrics
        """
        if phi_target is None:
            phi_target = h

        # Ensure on correct device (use W_local's device, not self.device,
        # because .to(cuda) moves buffers but doesn't update self.device)
        actual_device = self.W_local.device
        h = h.to(actual_device)
        phi_target = phi_target.to(actual_device)

        # Get self-calibrated parameters
        L_rw = compute_laplacian(self.W_local)
        precision = self.precision_tracker.get_precision()
        beta = self.beta
        n_steps = self.n_settle

        # === SETTLE ===
        # Auto-select method: SSM (spectral) when eigenmodes are
        # available and use_spectral_settling is True, else symplectic.
        # SSM is O(K) per settle instead of O(N^2) per step.
        phi = h.clone()
        if (
            self.config.use_spectral_settling
            and self.eigenvalues is not None
            and self.eigenvalues.abs().sum() > 0
        ):
            method = 'ssm'
            truncation_K = self.config.spectral_K if self.config.spectral_K > 0 else 0
        else:
            method = 'symplectic'
            truncation_K = 0

        phi, phi_dot, settle_info = settle(
            phi=phi,
            phi_target=phi_target,
            W=self.W_local,
            L_rw=L_rw,
            precision=precision,
            beta=beta,
            n_steps=n_steps,
            config=self.config,
            gate=self.gate,
            training=training,
            M_vector=self.mass,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            method=method,
            truncation_K=truncation_K,
        )

        # Mix with residual
        output = self.config.residual_weight * h + (1 - self.config.residual_weight) * phi

        # Initialize diagnostics dict early (needed by all paths)
        diagnostics = {}

        # === LOCAL LEARNING (no backprop!) ===
        stress_diag = {}
        geom_diag = {}
        error_mag = 0.0
        if training:
            with torch.no_grad():
                # NaN guard: skip local learning if phi is corrupted
                if torch.isnan(phi).any() or torch.isnan(self.W_local).any():
                    # Recover: reset W_local if it's NaN
                    if torch.isnan(self.W_local).any():
                        W_fresh, mask_fresh = initialize_W(
                            self.config.hidden_dim, self.config.sparsity, self.device
                        )
                        self.W_local.copy_(W_fresh)
                        self.mask.copy_(mask_fresh)
                    diagnostics[f'part_{self.part_id}_nan_recovery'] = True
                else:
                    # Update precision tracker
                    error = phi - phi_target
                    self.precision_tracker.update(error)
                    error_mag = error.detach().abs().mean().item()

                    # Self-calibrate
                    self._calibrate(error_mag)

                    # Compute stress tensor
                    S, stress_diag = compute_stress_tensor(
                        phi, phi_target, self.config, phi_dot
                    )

                    # Apply stress momentum (Titans)
                    S_smooth = self.stress_momentum.update(S)

                    # Compute kinetic stress
                    h_idx = (self.W_local ** 2).sum(dim=1).clamp(min=1e-8)
                    K_kinetic = compute_kinetic_stress(
                        self.W_local, self.mass, h_idx, phi_dot
                    )

                    # Data-dependent forgetting
                    forget_alpha = 0.0
                    if self.config.data_dependent_forgetting and self._prev_stress is not None:
                        forget_alpha = compute_data_dependent_alpha(S, self._prev_stress)
                    self._prev_stress = S.clone()

                    # Geometry step (update W_local)
                    self.W_local, geom_diag = geometry_step(
                        W=self.W_local,
                        S=S_smooth,
                        eta=self.eta,
                        config=self.config,
                        mask=self.mask,
                        K_kinetic=K_kinetic,
                        ei_mask=self.ei_mask,
                        forget_alpha=forget_alpha,
                    )

                    # Safety: clamp W_local to prevent runaway
                    self.W_local.clamp_(-5.0, 5.0)

                    # NaN guard after geometry step
                    if torch.isnan(self.W_local).any():
                        W_fresh, mask_fresh = initialize_W(
                            self.config.hidden_dim, self.config.sparsity, self.device
                        )
                        self.W_local.copy_(W_fresh)
                        self.mask.copy_(mask_fresh)
                        diagnostics[f'part_{self.part_id}_nan_recovery'] = True

                    # Structural plasticity (periodic)
                    self.global_step += 1
                    if self.global_step % self.config.plasticity_interval == 0:
                        self.W_local, self.mask = structural_plasticity(
                            self.W_local, self.mask, S,
                            target_sparsity=self.config.sparsity,
                        )

                    # Update eigendecomposition (periodic)
                    if self.global_step % self.config.eigen_update_interval == 0:
                        self._update_eigen()

                    # Mass-conductivity constraint
                    self.mass, mass_diag = mass_conductivity_constraint(
                        self.mass, self.W_local, self.config.alpha_1
                    )

                    # Update mass from Herfindahl
                    new_mass = self._compute_initial_mass(self.W_local)
                    self.mass = 0.95 * self.mass + 0.05 * new_mass  # Slow blend
                    self.mass.clamp_(0.1, 10.0)  # Safety bounds

                    # Astrocyte modulation (if enabled)
                    if self.config.astrocyte_enabled:
                        surprise = error_mag / (self.error_running_avg.item() + 1e-8)
                        self.calcium = (
                            (1 - 1.0/self.config.astrocyte_tau) * self.calcium
                            + (1.0/self.config.astrocyte_tau) * surprise
                        )

                    # Entropy tracking
                    entropy_diag = self.entropy_tracker.record(phi)

        # === SAFETY: Final NaN guard on output ===
        if torch.isnan(output).any():
            output = h.clone()  # Fall back to input if output is corrupted

        # === DIAGNOSTICS ===
        diagnostics.update({
            f'part_{self.part_id}_settle_steps': n_steps,
            f'part_{self.part_id}_arousal': self.arousal.item() if not torch.isnan(self.arousal) else 0.5,
            f'part_{self.part_id}_eta': self.eta,
            f'part_{self.part_id}_velocity_norm': phi_dot.detach().abs().mean().item() if not torch.isnan(phi_dot).any() else 0.0,
            f'part_{self.part_id}_phi_magnitude': phi.detach().abs().mean().item() if not torch.isnan(phi).any() else 0.0,
            **settle_info,
        })

        if training:
            diagnostics.update({
                f'part_{self.part_id}_stress_mean': stress_diag.get('stress_mean', 0),
                f'part_{self.part_id}_error_magnitude': error_mag,
                **{f'part_{self.part_id}_{k}': v for k, v in geom_diag.items()},
            })

        return output, phi_dot, diagnostics

    def grow(self, new_n_nodes: int):
        """
        Neurogenesis: Expand the brain's capacity while preserving learned structure.

        Implements the "Gestation" phase from theory/23_SOMI_LIFE_CYCLE.md:
        1. Create a larger container (new_n_nodes)
        2. Graft the old brain into the top-left corner
        3. Initialize new connections sparsely
        4. Recalibrate physics for the new spectrum

        This solves the local learning cold-start problem (theory/24_GROWTH_SOLVES_LOCAL_LEARNING.md):
        new neurons connect to already-organized geometry, so local stress signals
        are strong enough for Hebbian learning to work.

        Args:
            new_n_nodes: Target size (must be > current hidden_dim)
        """
        old_H = self.W_local.shape[0]
        if new_n_nodes <= old_H:
            return

        device = self.W_local.device

        # --- W_local and mask: graft old into top-left of new ---
        # initialize_W uses CPU-only ops, so create on CPU then move
        W_new, mask_new = initialize_W(new_n_nodes, self.config.sparsity, torch.device('cpu'))
        with torch.no_grad():
            W_new[:old_H, :old_H] = self.W_local.cpu()
            mask_new[:old_H, :old_H] = self.mask.cpu()
        self.register_buffer('W_local', W_new.to(device))
        self.register_buffer('mask', mask_new.to(device))

        # --- Mass: preserve old, compute new from Herfindahl ---
        mass_new = self._compute_initial_mass(self.W_local)
        with torch.no_grad():
            mass_new[:old_H] = self.mass[:old_H]
        self.register_buffer('mass', mass_new)

        # --- Dale's Law E/I mask ---
        if self.ei_mask is not None:
            n_exc = int(new_n_nodes * self.config.ei_ratio)
            ei_new = torch.ones(new_n_nodes, device=device)
            ei_new[n_exc:] = -1.0
            self.register_buffer('ei_mask', ei_new)

        # --- PrecisionTracker: preserve old EMA state ---
        old_precision_var = self.precision_tracker.error_variance
        self.precision_tracker = PrecisionTracker(new_n_nodes, self.config.precision_ema).to(device)
        with torch.no_grad():
            self.precision_tracker.error_variance[:old_H] = old_precision_var

        # --- BasalGangliaGate: stateless, just resize ---
        self.gate = BasalGangliaGate(new_n_nodes, self.config.gate_bottleneck).to(device)

        # --- StressMomentum: preserve old EMA ---
        old_S_ema = self.stress_momentum.S_ema
        self.stress_momentum = StressMomentum(new_n_nodes, self.config.stress_momentum_beta, device)
        with torch.no_grad():
            if hasattr(self.stress_momentum, 'S_ema') and old_S_ema is not None:
                S_template = torch.zeros(new_n_nodes, new_n_nodes, device=device)
                S_template[:old_H, :old_H] = old_S_ema
                self.stress_momentum.S_ema = S_template
                self.stress_momentum.initialized = True

        # --- Eigendecomposition cache: resize and recompute ---
        self.register_buffer('eigenvalues', torch.zeros(new_n_nodes, device=device))
        self.register_buffer('eigenvectors', torch.eye(new_n_nodes, device=device))
        # _update_eigen uses scatter_ which needs matching devices;
        # run on CPU then move back if on GPU
        if device.type != 'cpu':
            self.W_local = self.W_local.cpu()
            self.eigenvalues = self.eigenvalues.cpu()
            self.eigenvectors = self.eigenvectors.cpu()
            self._update_eigen()
            self.register_buffer('W_local', self.W_local.to(device))
            self.register_buffer('eigenvalues', self.eigenvalues.to(device))
            self.register_buffer('eigenvectors', self.eigenvectors.to(device))
        else:
            self._update_eigen()

        # --- Conduction delays: resize if enabled ---
        if self.config.delays_enabled and hasattr(self, 'delays'):
            delays_new = 1.0 / (self.W_local.abs().clamp(min=1e-4))
            delays_new = delays_new.clamp(*self.config.delay_range)
            self.register_buffer('delays', delays_new)

        # --- Reset previous stress (will be recomputed on next forward) ---
        self._prev_stress = None
