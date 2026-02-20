"""
SOMI 2.0 Geometry Equation
============================

The geometry W learns from the field's activity and errors:

    Ẇ = −η ∇_W V − λ_W W

This is gradient flow on the potential — geometry descends the energy
landscape just like the field does, but on a slower timescale.

The information stress tensor S_ij tells each connection how to change:
  - High activity mismatch between connected features → weaken connection
  - High error mismatch between connected features → weaken connection
  - High activity correlation between connected features → strengthen connection
  - STDP temporal correlation (phi_i * phi_dot_j) → directional strengthening

Brain analog:
  - Anti-Hebbian: different activations → disconnect (like lateral inhibition)
  - Hebbian: fire together → wire together (classical Hebb's rule)
  - STDP: pre-leads-post → strengthen forward connection (directional wiring)
  - Metabolic decay: unused connections die (synaptic pruning)

After each update, W is constrained to be:
  - ASYMMETRIC (directed connections — W_ij != W_ji, like real synapses)
  - Non-negative (excitatory)
  - Zero diagonal (no self-connections)
  - Soft row cap + soft column cap (homeostatic)
  - Optionally sparse (structural plasticity — grow/prune connections)

Source: SOMI_2_0_UNIFIED_ACTION.md, extended with asymmetric W (Feb 2026)
"""

import torch
from typing import Tuple, Optional

from ..config import SOMIBrainConfig as SOMIConfig


def compute_stress_tensor(
    phi: torch.Tensor,         # [batch, seq, hidden]
    phi_target: torch.Tensor,  # [batch, seq, hidden]
    config: SOMIConfig,
    phi_dot: Optional[torch.Tensor] = None,  # [batch, seq, hidden] velocity
) -> torch.Tensor:
    """
    Compute the information stress tensor S_ij.

    The stress tensor is like Einstein's stress-energy tensor in General
    Relativity — it tells the geometry how to curve (learn). In GR,
    matter tells space how to curve. In SOMI, information tells the
    connectivity how to change.

    Formula:
        S_ij = (alpha_1/2)<(phi_i - phi_j)^2>
             + (lambda_E/2)<(e_i - e_j)^2>
             - (lambda_C/4)[<phi_i phi_j> + kappa_stdp * <phi_i phi_dot_j>]

    Four components:
      1. Activity mismatch (anti-Hebbian from coupling):
         If connected features have DIFFERENT activations, stress is HIGH
         -> weaken the connection (they shouldn't be coupled)

      2. Error mismatch (anti-Hebbian from error smoothing):
         If connected features have DIFFERENT errors, stress is HIGH
         -> weaken the connection (errors shouldn't spread between them)

      3. Activity correlation (Hebbian from coordination):
         If connected features have SIMILAR activations, stress is LOW
         -> strengthen the connection (they should stay coupled)

      4. STDP temporal correlation (directional Hebbian):
         If feature i is active when feature j is CHANGING (phi_dot_j > 0),
         strengthen i<-j (j drives i). This creates directed pathways.
         The kappa_stdp parameter controls the strength.

         From the action: adding gamma/2 * phi^T W phi_dot to the
         Lagrangian produces this term in delta_S/delta_W_ij.
         ChatGPT verified (Feb 11, 2026) that this is action-consistent
         and produces a gyroscopic force (W-W^T)*phi_dot in the field
         equation — does no work, just rotates trajectories.

    Args:
        phi:        Settled field activations [batch, seq, hidden]
        phi_target: Target activations [batch, seq, hidden]
        config:     SOMI configuration
        phi_dot:    Field velocity [batch, seq, hidden] (optional, for STDP)

    Returns:
        S: [hidden, hidden] stress tensor
    """
    hidden_dim = phi.shape[-1]
    error = phi - phi_target

    # Flatten batch and sequence into one dimension
    phi_flat = phi.detach().reshape(-1, hidden_dim)   # [N, H]
    err_flat = error.detach().reshape(-1, hidden_dim)  # [N, H]
    N = phi_flat.shape[0]

    # === INPUT NORMALIZATION ===
    # Per-SAMPLE normalization: each batch element gets unit norm so it
    # contributes equally to the correlation. Makes C_ij bounded in [-1, 1].
    phi_norm = phi_flat - phi_flat.mean(dim=0, keepdim=True)
    phi_scale = phi_norm.norm(dim=1, keepdim=True) + 1e-8  # [batch, 1]
    phi_norm = phi_norm / phi_scale

    err_norm = err_flat - err_flat.mean(dim=0, keepdim=True)
    err_scale = err_norm.norm(dim=1, keepdim=True) + 1e-8  # [batch, 1]
    err_norm = err_norm / err_scale

    # === Activity correlation matrix: C_ij = <phi_i phi_j> ===
    C = (phi_norm.T @ phi_norm) / N  # [H, H]

    # === Activity mismatch: A_ij = <(phi_i - phi_j)^2> ===
    phi_sq = (phi_norm ** 2).mean(dim=0)  # [H]
    A = phi_sq.unsqueeze(1) + phi_sq.unsqueeze(0) - 2 * C  # [H, H]

    # === Error mismatch: E_ij = <(e_i - e_j)^2> ===
    E_corr = (err_norm.T @ err_norm) / N  # [H, H]
    e_sq = (err_norm ** 2).mean(dim=0)  # [H]
    E = e_sq.unsqueeze(1) + e_sq.unsqueeze(0) - 2 * E_corr  # [H, H]

    # === STDP temporal correlation: T_ij = <phi_i phi_dot_j> ===
    # This is the DIRECTIONAL learning signal. It's ASYMMETRIC:
    #   T_ij != T_ji (phi_i * phi_dot_j != phi_j * phi_dot_i)
    # When feature j is changing fast (phi_dot_j large) while feature i
    # is active (phi_i large), T_ij is large -> strengthen j->i connection.
    # This creates feedforward chains from leaders to followers.
    T = None
    if phi_dot is not None and config.kappa_stdp > 0:
        pdot_flat = phi_dot.detach().reshape(-1, hidden_dim)  # [N, H]

        # Normalize phi_dot the same way as phi (per-sample unit norm)
        # This keeps the STDP signal on the same scale as the correlation C
        pdot_norm = pdot_flat - pdot_flat.mean(dim=0, keepdim=True)
        pdot_scale = pdot_norm.norm(dim=1, keepdim=True) + 1e-8
        pdot_norm = pdot_norm / pdot_scale

        # T_ij = <phi_i * phi_dot_j> — NOTE: this is NOT symmetric!
        # T[i,j] = how much feature i's activity correlates with
        #          feature j's RATE OF CHANGE
        # phi_norm is [N, H], pdot_norm is [N, H]
        # T = phi_norm^T @ pdot_norm / N  -> [H, H]
        T = (phi_norm.T @ pdot_norm) / N  # [H, H] — ASYMMETRIC

    # === Stress tensor ===
    # Positive terms (anti-Hebbian): mismatch -> weaken
    # Negative terms (Hebbian + STDP): correlation -> strengthen
    S = (
        0.5 * config.alpha_1 * A         # Activity mismatch
        + 0.5 * config.lambda_E * E      # Error mismatch
        - 0.5 * config.lambda_C * C      # Activity correlation (Hebbian)
    )

    # Add STDP term (from the action's coordination potential)
    # The factor of 0.5 * lambda_C matches the derivation:
    #   dV_coord/dW_ij = -lambda_C/4 * (phi_i*phi_j + kappa*phi_i*phi_dot_j)
    # Since we already have -0.5*lambda_C*C for the phi_i*phi_j part,
    # the STDP part is -0.5*lambda_C*kappa*T
    if T is not None:
        S = S - 0.5 * config.lambda_C * config.kappa_stdp * T

    return S


def compute_kinetic_stress(
    W: torch.Tensor,         # [hidden, hidden]
    M_vector: torch.Tensor,  # [hidden] per-feature mass
    h: torch.Tensor,         # [hidden] Herfindahl index
    phi_dot: torch.Tensor,   # [batch, seq, hidden] velocity
) -> torch.Tensor:
    """
    Compute kinetic stress from geometry-dependent inertia.

    THIS IS NEW IN SOMI 2.0 COMPLETE THEORY.

    When the metric tensor G(W) depends on W, varying the action
    with respect to W produces an extra term in the geometry equation.
    This is exactly analogous to the kinetic term in Einstein's field
    equations: matter in motion curves spacetime differently than matter
    at rest.

    Formula (diagonal approximation):
        K_ij^kinetic = -(M_i / h_i) * W_ij * <phi_dot_i^2>

    where:
        M_i = per-feature mass
        h_i = Herfindahl index (connection concentration)
        W_ij = current connection weight
        <phi_dot_i^2> = time-averaged squared velocity (kinetic energy)

    Physical meaning:
        - If feature i is oscillating fast (high phi_dot_i^2) and connected
          to feature j via W_ij, this STRENGTHENS the connection (K is negative
          stress, so negative K means "good connection, strengthen it")
        - Features with high kinetic energy attract connections
        - This is a self-reinforcing mechanism: connections that carry
          dynamic information are strengthened

    Brain analog:
        - Activity-dependent myelination: axons that carry more traffic
          get thicker myelin sheaths (faster conduction)
        - Hebbian plasticity driven by DYNAMICS, not just static activity

    Validated by:
        - Einstein field equations (stress-energy tensor includes kinetic terms)
        - Predictive coding waves (Faye et al. 2025): propagating waves
          preferentially strengthen pathways they travel along

    Args:
        W:        [hidden, hidden] connectivity matrix
        M_vector: [hidden] per-feature mass
        h:        [hidden] Herfindahl index
        phi_dot:  [batch, seq, hidden] field velocity

    Returns:
        K_kinetic: [hidden, hidden] kinetic stress tensor
    """
    # Mean squared velocity per feature (averaged over batch and sequence)
    # This is <phi_dot_i^2>, proportional to per-feature kinetic energy
    kinetic_energy = (phi_dot.detach() ** 2).mean(dim=(0, 1))  # [hidden]

    # K_ij^kinetic = -(M_i / h_i) * W_ij * <phi_dot_i^2>
    # The minus sign means kinetic energy REDUCES stress (strengthens connections)
    prefactor = M_vector / (h + 1e-8)  # [hidden]
    K_kinetic = -(prefactor * kinetic_energy).unsqueeze(-1) * W  # [hidden, hidden]

    # NO symmetrization — K_kinetic is naturally asymmetric:
    # K_ij depends on feature i's velocity and mass, not j's.
    # This means features with high kinetic energy preferentially
    # strengthen their OUTGOING connections (activity-dependent myelination).

    return K_kinetic


def geometry_step(
    W: torch.Tensor,                      # [hidden, hidden]
    S: torch.Tensor,                      # [hidden, hidden]
    eta: float,                           # Learning rate (from calibration)
    config: SOMIConfig,
    mask: Optional[torch.Tensor] = None,  # [hidden, hidden] sparsity mask
    K_kinetic: Optional[torch.Tensor] = None,  # [hidden, hidden] kinetic stress
) -> torch.Tensor:
    """
    One geometry update step.

    Update rule (SOMI 2.0 Complete Theory):
        dW = -eta * (S + K_kinetic) - lambda_W * W

    The stress S comes from the potential V (static information stress).
    K_kinetic comes from the metric tensor's dependence on W (dynamic stress).
    Together they form the COMPLETE geometry equation from the action.

    Then enforce constraints (non-negative, row/col caps, sparsity, etc.)

    The first term (-eta S) is learning from information mismatch.
    The second term (-eta K_kinetic) is learning from dynamics (new in 2.0).
    The third term (-lambda_W W) is decay: unused connections slowly weaken.

    Brain analog:
      - S: Hebbian/anti-Hebbian plasticity from correlations
      - K_kinetic: Activity-dependent myelination from dynamics
      - Decay: Metabolic cost means unused synapses are pruned during sleep

    Args:
        W:         Current connectivity matrix
        S:         Information stress tensor (from compute_stress_tensor)
        eta:       Geometry learning rate (from calibration, arousal-modulated)
        config:    SOMI configuration
        mask:      Sparsity mask (1 = connection exists, 0 = no connection)
        K_kinetic: Kinetic stress tensor (from compute_kinetic_stress). None = no kinetic stress.

    Returns:
        W_new: Updated connectivity matrix (new tensor, does not modify input)
    """
    with torch.no_grad():
        # Total stress = information stress + kinetic stress
        total_stress = S
        if K_kinetic is not None:
            total_stress = total_stress + K_kinetic

        # Compute update
        dW = -eta * total_stress - config.lambda_W * W

        # Apply sparsity mask (only update existing connections)
        if mask is not None:
            dW = dW * mask

        # Apply update
        W_new = W + dW

        # Enforce all constraints
        W_new = enforce_constraints(W_new, mask)

    return W_new


def enforce_constraints(
    W: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sinkhorn_iters: int = 0,
) -> torch.Tensor:
    """
    Enforce W constraints after each geometry update.

    ASYMMETRIC W (Feb 11, 2026):
      W is now DIRECTED: W_ij != W_ji. Real synapses are unidirectional.
      The asymmetric part enables feedforward routing (sensory→motor).

      Constraints:
        1. NO symmetrization (directed connections)
        2. Zero diagonal (no self-connections)
        3. Non-negative (Dale's Law for excitatory connections)
        4. Sparsity mask (structural constraint)
        5. Soft row cap at 2.0 (prevent excessive incoming weight)
        6. Soft column cap at 2.0 (prevent excessive outgoing weight)

      Stability: ChatGPT verified (Feb 11, 2026) that with row-stochastic
      P = D_out^{-1} W, stability requires beta^2 > 2*alpha*m. Our damping
      and anchoring satisfy this. The soft row/column caps ensure W stays
      bounded even without symmetrization.
    """
    with torch.no_grad():
        # 1. NO symmetrization — W is now directed (W_ij != W_ji)
        #    Real synapses are unidirectional (axon→dendrite).
        #    The antisymmetric part (W - W^T)/2 creates directed
        #    information flow: sensory → association → motor.

        # 2. Zero diagonal
        W.fill_diagonal_(0)

        # 3. Non-negative
        W = W.clamp(min=0)

        # 4. Sparsity mask
        if mask is not None:
            W = W * mask

        # 5. Soft row cap: prevent any feature from receiving too much input
        row_sums = W.sum(dim=1, keepdim=True)
        cap = 2.0
        scale = torch.where(
            row_sums > cap,
            cap / row_sums.clamp(min=1e-8),
            torch.ones_like(row_sums)
        )
        W = W * scale

        # 6. Soft column cap: prevent any feature from sending too much output
        #    (new for asymmetric W — without this, a single feature could
        #    dominate as a "broadcaster" with huge outgoing weights)
        col_sums = W.sum(dim=0, keepdim=True)
        col_scale = torch.where(
            col_sums > cap,
            cap / col_sums.clamp(min=1e-8),
            torch.ones_like(col_sums)
        )
        W = W * col_scale

    return W


def sinkhorn_normalize(W: torch.Tensor, n_iters: int = 10) -> torch.Tensor:
    """
    Make W doubly stochastic via Sinkhorn-Knopp algorithm.

    The algorithm alternates between normalizing rows and columns:
      Step 1: Divide each row by its sum (now all rows sum to 1)
      Step 2: Divide each column by its sum (now all columns sum to 1,
              but rows don't anymore)
      Step 3: Repeat — the matrix converges to doubly stochastic

    After convergence: every row sums to 1, every column sums to 1.

    This is like a room full of people who each need exactly 1 unit of
    attention total. Sinkhorn ensures everyone gives and receives exactly
    the right total amount.

    Args:
        W: Non-negative matrix to normalize
        n_iters: Number of alternating normalization steps

    Returns:
        W: Doubly stochastic matrix
    """
    for _ in range(n_iters):
        # Row normalize
        row_sums = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
        W = W / row_sums

        # Column normalize
        col_sums = W.sum(dim=0, keepdim=True).clamp(min=1e-8)
        W = W / col_sums

    return W


def initialize_W(
    hidden_dim: int,
    sparsity: float = 0.1,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize ASYMMETRIC connectivity matrix W and sparsity mask.

    Creates a sparse, DIRECTED connectivity matrix. The mask is symmetric
    (if i→j exists, j→i also exists — both directions are possible) but
    the WEIGHTS are independent (W_ij != W_ji). This means:
      - Every pair that CAN connect has both directions available
      - The stress tensor + STDP will learn which direction is stronger
      - Sensory→motor pathways can emerge from initially symmetric topology

    Brain analog: Cortical connectivity is ~10%. At birth, many bidirectional
    connections exist. Experience (STDP) prunes and strengthens specific
    directions — creating feedforward sensory→motor pathways while keeping
    some feedback (motor→sensory for prediction).

    Args:
        hidden_dim: Number of features
        sparsity:   Fraction of possible connections to activate (default 10%)
        device:     Torch device

    Returns:
        W:    [hidden_dim, hidden_dim] directed connectivity matrix
        mask: [hidden_dim, hidden_dim] binary sparsity mask (symmetric)
    """
    if device is None:
        device = torch.device('cpu')

    # Number of possible connections (upper triangle, excluding diagonal)
    n_possible = hidden_dim * (hidden_dim - 1) // 2
    n_connections = max(hidden_dim, int(n_possible * sparsity))  # At least hidden_dim

    # Create sparse mask (symmetric — both directions are possible)
    mask = torch.zeros(hidden_dim, hidden_dim, device=device)

    # Random upper triangle indices
    triu_indices = torch.triu_indices(hidden_dim, hidden_dim, offset=1)
    n_available = triu_indices.shape[1]

    # Select random connections
    n_to_select = min(n_connections, n_available)
    selected = torch.randperm(n_available, device=device)[:n_to_select]

    rows = triu_indices[0, selected]
    cols = triu_indices[1, selected]

    mask[rows, cols] = 1.0
    mask = mask + mask.T  # Symmetric mask (both directions available)

    # Initialize weights: INDEPENDENT random values for each direction
    # No symmetrization — W_ij and W_ji are independent from the start
    W = torch.rand(hidden_dim, hidden_dim, device=device) * mask
    W.fill_diagonal_(0)       # No self-connections

    # Simple row normalization (each row sums to ~1.0 initially)
    row_sums = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
    W = W / row_sums

    return W, mask


def structural_plasticity(
    W: torch.Tensor,
    mask: torch.Tensor,
    S: torch.Tensor,
    target_sparsity: float = 0.1,
    turnover_rate: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prune weak connections and grow needed ones.

    This is structural plasticity — not just changing connection STRENGTHS
    (that's the geometry equation), but changing which connections EXIST
    (adding new ones, removing old ones).

    Process:
      1. Find the weakest existing connections → prune them (set mask to 0)
      2. Find the locations with highest stress but no connection → grow them
      3. Keep total connection count approximately constant

    Brain analog:
      - Synaptic pruning: Unused/weak synapses are physically removed
        (happens heavily during sleep and adolescence)
      - Synaptogenesis: New synapses form where they're needed
        (happens throughout life in response to learning)

    Source: Frontiers 2025, BCPNN 2025 (Doc 28)

    Args:
        W:               Current connectivity [hidden, hidden]
        mask:            Current sparsity mask [hidden, hidden]
        S:               Stress tensor [hidden, hidden]
        target_sparsity: Target connectivity fraction
        turnover_rate:   Fraction of connections to prune/grow per event

    Returns:
        W_new:    Updated connectivity (pruned connections zeroed out)
        mask_new: Updated sparsity mask
    """
    hidden_dim = W.shape[0]

    with torch.no_grad():
        # Work with FULL matrix (asymmetric W — each direction is independent)
        # The mask is symmetric (both directions possible), but we prune/grow
        # individual DIRECTED edges based on their individual weights/stress.

        # Off-diagonal mask entries (exclude self-connections)
        offdiag = mask.clone()
        offdiag.fill_diagonal_(0)
        n_existing = int(offdiag.sum().item())
        n_turnover = max(1, int(n_existing * turnover_rate))

        # === PRUNE: Remove weakest existing DIRECTED connections ===
        existing_weights = (W * offdiag).view(-1)
        existing_bool = offdiag.view(-1).bool()
        existing_values = existing_weights[existing_bool]

        if len(existing_values) > n_turnover:
            # Find the n_turnover weakest directed connections
            _, prune_local = existing_values.topk(n_turnover, largest=False)
            prune_global = torch.where(existing_bool)[0][prune_local]

            # Remove from mask (just this direction, not the mirror)
            flat_mask = mask.view(-1).clone()
            flat_mask[prune_global] = 0
            mask = flat_mask.reshape(hidden_dim, hidden_dim)

        # === GROW: Add directed connections based on FUNCTIONAL NEED ===
        # S_ij measures the stress for the directed edge i←j:
        #   - Negative S_ij = correlation-dominated → should connect
        #   - We grow directed edges independently
        absent = (1 - mask).clone()
        absent.fill_diagonal_(0)  # Never grow self-connections

        neg_stress = (-S * absent).view(-1)
        abs_stress = (S.abs() * absent).view(-1)
        growth_signal = (0.7 * neg_stress + 0.3 * abs_stress)
        absent_bool = absent.view(-1).bool()
        absent_values = growth_signal[absent_bool]

        if len(absent_values) > n_turnover:
            _, grow_local = absent_values.topk(
                min(n_turnover, len(absent_values)), largest=True
            )
            grow_global = torch.where(absent_bool)[0][grow_local]

            # Add directed edge to mask (just this direction)
            flat_mask = mask.view(-1)
            flat_mask[grow_global] = 1
            mask = flat_mask.reshape(hidden_dim, hidden_dim)

        # Clean up
        mask.fill_diagonal_(0)

        # Zero out weights for pruned connections
        W = W * mask

        # Initialize new connections with small random weights
        # No symmetrization — each direction gets its own independent weight
        new_connections = (mask > 0) & (W == 0)
        if new_connections.any():
            W[new_connections] = 0.01 * torch.rand(
                new_connections.sum(), device=W.device
            )
            W.fill_diagonal_(0)

        # Soft re-normalize (row + column caps, no Sinkhorn)
        W = W * mask
        row_sums = W.sum(dim=1, keepdim=True)
        cap = 2.0
        scale = torch.where(row_sums > cap, cap / row_sums.clamp(min=1e-8),
                            torch.ones_like(row_sums))
        W = W * scale

        col_sums = W.sum(dim=0, keepdim=True)
        col_scale = torch.where(col_sums > cap, cap / col_sums.clamp(min=1e-8),
                                torch.ones_like(col_sums))
        W = W * col_scale

    return W, mask
