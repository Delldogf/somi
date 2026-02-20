"""
SOMI Lossless Spectral Absorption
====================================

Complete pipeline for absorbing transformer knowledge into SOMI
with near-zero information loss. Uses:

  Phase 2: Marchenko-Pastur noise removal (not ad-hoc SVD truncation)
  Phase 3: Procrustes alignment across models
  Phase 4: Dynamic SOMI growth to match effective rank
  Phase 5: Kalman-filtered multi-model fusion
  Phase 6: Full-strength spectral installation
  Phase 7: Full-rank vocabulary transfer

Theory: SOMI_Research/SOMI_4/MATHEMATICAL_TOOLS_FOR_ABSORPTION.md
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple

from .spectral_analysis import marchenko_pastur_threshold
from .alignment import svd_align


# ---------------------------------------------------------------------------
# Phase 2: Lossless Spectral Extraction
# ---------------------------------------------------------------------------

def lossless_spectral_extract(
    layer_data: Dict,
    hidden_dim: int,
    device: torch.device = torch.device('cpu'),
) -> Optional[Dict]:
    """
    Extract the full spectral content of a transformer layer using
    Marchenko-Pastur thresholding to separate signal from noise.

    Instead of deriving a single connectivity matrix (lossy), we extract:
    - Full eigenvalues of the combined attn+MLP correlation
    - Full eigenvectors (coordinate directions)
    - MP-filtered signal modes only (noise removed)
    - Mass vector from LayerNorm

    Returns None if extraction fails for this layer.
    """
    attn_out = layer_data.get('attn_out')
    attn_qkv = layer_data.get('attn_qkv')
    mlp_down = layer_data.get('mlp_down')
    mlp_up = layer_data.get('mlp_up')

    C_attn = None
    C_mlp = None

    # Attention correlation: C_attn = W_out^T @ W_out
    if attn_out is not None:
        a = attn_out.float().to(device)
        if a.shape[0] == hidden_dim:
            C_attn = a.T @ a
        elif a.shape[-1] == hidden_dim:
            C_attn = a @ a.T

    # MLP transformation: C_mlp = W_down @ W_up
    if mlp_down is not None and mlp_up is not None:
        d = mlp_down.float().to(device)
        u = mlp_up.float().to(device)
        try:
            if d.shape[0] == hidden_dim and u.shape[-1] == hidden_dim:
                C_mlp = d @ u
            elif d.shape[-1] == hidden_dim and u.shape[0] == hidden_dim:
                C_mlp = d.T @ u.T
            elif d.shape[-1] == u.shape[0]:
                C_mlp = d @ u
                if C_mlp.shape[0] != hidden_dim:
                    C_mlp = None
            elif d.shape[0] == u.shape[-1]:
                C_mlp = d.T @ u.T
                if C_mlp.shape[0] != hidden_dim:
                    C_mlp = None
        except RuntimeError:
            C_mlp = None

    # Combine available correlations
    if C_attn is not None and C_mlp is not None:
        C = C_attn + C_mlp
    elif C_attn is not None:
        C = C_attn
    elif C_mlp is not None:
        C = C_mlp
    else:
        return None

    # Symmetrize for real eigendecomposition
    C_sym = 0.5 * (C + C.T)

    # Full eigendecomposition (ascending order)
    eigenvalues, eigenvectors = torch.linalg.eigh(C_sym)

    # Apply Marchenko-Pastur threshold
    mp = marchenko_pastur_threshold(C_sym)
    K_signal = mp['K_signal']

    # Keep only signal modes (highest eigenvalues are at the end for eigh)
    if K_signal > 0:
        signal_eigenvalues = eigenvalues[-K_signal:]
        signal_eigenvectors = eigenvectors[:, -K_signal:]
    else:
        signal_eigenvalues = eigenvalues
        signal_eigenvectors = eigenvectors

    # Mass from LayerNorm
    mass = None
    if layer_data.get('ln_weight') is not None:
        ln = layer_data['ln_weight'].float().to(device)
        importance = ln.abs()
        mass = (importance / importance.mean().clamp(min=1e-8)).clamp(0.1, 10.0)

    return {
        'eigenvalues': signal_eigenvalues,
        'eigenvectors': signal_eigenvectors,
        'all_eigenvalues': eigenvalues,
        'K_signal': K_signal,
        'K_total': eigenvalues.shape[0],
        'energy_in_signal': mp['energy_in_signal'],
        'effective_rank': mp['effective_rank'],
        'mass': mass,
    }


# ---------------------------------------------------------------------------
# Phase 3: Procrustes Alignment
# ---------------------------------------------------------------------------

def procrustes_align(
    V_reference: torch.Tensor,
    V_source: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Orthogonal Procrustes alignment — find the rotation R that best
    aligns V_source to V_reference.

    Closed-form solution: R = U @ V^T from SVD(V_ref^T @ V_src)

    This preserves distances and angles (geometry), unlike SVD truncation
    which distorts the space.

    Args:
        V_reference: [dim, K_ref] reference eigenvectors
        V_source:    [dim, K_src] source eigenvectors to align

    Returns:
        V_aligned: [dim, K] aligned eigenvectors
        R: [K, K] rotation matrix
        alignment_error: scalar, lower is better
    """
    K = min(V_reference.shape[1], V_source.shape[1])
    dim = min(V_reference.shape[0], V_source.shape[0])

    ref = V_reference[:dim, :K].float()
    src = V_source[:dim, :K].float()

    # Center both
    ref_centered = ref - ref.mean(dim=0, keepdim=True)
    src_centered = src - src.mean(dim=0, keepdim=True)

    # Cross-correlation matrix
    M = ref_centered.T @ src_centered  # [K, K]

    # SVD gives optimal rotation
    U, _, Vh = torch.linalg.svd(M)
    R = U @ Vh  # [K, K]

    # Apply rotation
    V_aligned = src_centered @ R.T  # [dim, K]

    # Measure alignment quality
    alignment_error = (V_aligned - ref_centered).norm() / ref_centered.norm().clamp(min=1e-8)

    return V_aligned, R, alignment_error.item()


# ---------------------------------------------------------------------------
# Phase 5: Kalman-Filtered Sequential Absorption
# ---------------------------------------------------------------------------

class KalmanSpectralFuser:
    """
    Kalman filter for optimal sequential fusion of spectral modes
    from multiple transformer models.

    For each eigenvalue mode, maintains:
    - x: current best estimate of the eigenvalue
    - P: uncertainty (covariance) of the estimate
    - eigenvector: accumulated aligned eigenvector

    Each new model's spectral content updates these optimally via
    the Kalman gain equation.
    """

    def __init__(self, n_modes: int, process_noise: float = 0.01):
        self.n_modes = n_modes
        self.x = torch.zeros(n_modes)           # eigenvalue estimates
        self.P = torch.ones(n_modes) * 100.0     # high initial uncertainty
        self.Q = process_noise                    # process noise
        self.v_sum = None                         # weighted eigenvector accumulator
        self.v_weight_sum = torch.zeros(n_modes)  # total weight per mode
        self.n_absorbed = 0
        self.history = []

    def absorb(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        measurement_noise: float = 1.0,
        model_weight: float = 1.0,
    ) -> Dict:
        """
        Absorb one model's spectral content using Kalman update.

        Args:
            eigenvalues: [K] signal eigenvalues from this model
            eigenvectors: [dim, K] corresponding eigenvectors (Procrustes-aligned)
            measurement_noise: R — how noisy this source is (higher = trust less)
            model_weight: extra weighting (e.g., proportional to model size)
        """
        K_eval = min(eigenvalues.shape[0], self.n_modes)
        K_evec = min(eigenvectors.shape[1], self.n_modes)
        K = min(K_eval, K_evec)
        device = eigenvalues.device

        # Pad eigenvalues to n_modes if needed
        z = torch.zeros(self.n_modes, device=device)
        z[:K] = eigenvalues[-K:]  # take the largest K eigenvalues

        # Measurement noise (higher for smaller/noisier models)
        R = torch.ones(self.n_modes, device=device) * measurement_noise
        R[K:] = 1e6  # infinite noise for missing modes

        # Kalman predict step (static model: x doesn't change)
        P_pred = self.P.to(device) + self.Q

        # Kalman gain: K = P / (P + R)
        K_gain = P_pred / (P_pred + R)

        # Update eigenvalue estimate
        self.x = self.x.to(device)
        innovation = z - self.x
        self.x = self.x + K_gain * innovation

        # Update uncertainty
        self.P = ((1 - K_gain) * P_pred).cpu()
        self.x = self.x.cpu()

        # Accumulate eigenvectors (weighted by Kalman gain * model_weight)
        dim = eigenvectors.shape[0]
        if self.v_sum is None:
            self.v_sum = torch.zeros(dim, self.n_modes)

        weight = (K_gain[:K] * model_weight).cpu()
        evecs_slice = eigenvectors[:, -K:].cpu()  # take last K columns (highest modes)
        dim_use = min(dim, self.v_sum.shape[0])
        self.v_sum[:dim_use, :K] += evecs_slice[:dim_use, :] * weight.unsqueeze(0)
        self.v_weight_sum[:K] += weight

        self.n_absorbed += 1
        avg_gain = K_gain[:K].mean().item()
        self.history.append({
            'model_idx': self.n_absorbed,
            'K_modes': K,
            'avg_kalman_gain': avg_gain,
            'avg_innovation': innovation[:K].abs().mean().item(),
        })

        return {
            'kalman_gain_avg': avg_gain,
            'innovation_avg': innovation[:K].abs().mean().item(),
            'uncertainty_avg': self.P[:K].mean().item(),
        }

    def get_fused_spectrum(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the Kalman-optimal fused eigenvalues and eigenvectors.

        Returns:
            eigenvalues: [n_modes] fused eigenvalues
            eigenvectors: [dim, n_modes] fused eigenvectors (normalized)
        """
        # Normalize accumulated eigenvectors
        V = self.v_sum.clone()
        for k in range(self.n_modes):
            w = self.v_weight_sum[k]
            if w > 1e-8:
                V[:, k] /= w
                # Re-normalize to unit length
                norm = V[:, k].norm()
                if norm > 1e-8:
                    V[:, k] /= norm

        return self.x.clone(), V


# ---------------------------------------------------------------------------
# Phase 6: Spectral Installation (Full Strength)
# ---------------------------------------------------------------------------

def install_spectrum_into_part(
    part,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    strength: float = 1.0,
) -> Dict:
    """
    Install fused spectral content into a SOMI Part's W_local.

    Reconstructs W from the eigenvector/eigenvalue pairs:
        W = V @ diag(lambda) @ V^T

    Unlike the old spectral_mode_transfer which uses the TARGET's
    eigenvectors, this uses the FUSED eigenvectors (which carry
    knowledge from all absorbed models via Kalman fusion).

    Args:
        part: SOMIPart to install into
        eigenvalues: [K] eigenvalues to install
        eigenvectors: [dim, K] eigenvectors to install
        strength: 1.0 for full replacement, <1.0 for blending
    """
    H = part.config.hidden_dim
    device = part.W_local.device
    K = min(eigenvalues.shape[0], eigenvectors.shape[1], H)
    evec_dim = eigenvectors.shape[0]

    with torch.no_grad():
        evals = eigenvalues[-K:].to(device).float()  # highest K modes
        # If eigenvectors are smaller than H, pad with zeros
        if evec_dim < H:
            padded = torch.zeros(H, K, device=device)
            padded[:evec_dim, :] = eigenvectors[:, -K:].to(device).float()
            evecs = padded
        else:
            evecs = eigenvectors[:H, -K:].to(device).float()

        # Reconstruct W from spectral content
        W_new = evecs @ torch.diag(evals.abs()) @ evecs.T

        # Enforce connectivity constraints
        W_new = W_new.abs()
        W_new.fill_diagonal_(0)
        row_sums = W_new.sum(1, keepdim=True).clamp(min=1e-8)
        W_new = W_new / row_sums

        # Install
        W_before = part.W_local.clone()
        part.W_local.copy_(
            (1 - strength) * part.W_local + strength * W_new
        )

        change = (part.W_local - W_before).abs().mean().item()

    # Update eigendecomposition cache
    part._update_eigen()

    return {
        'K_installed': K,
        'strength': strength,
        'W_change': change,
        'eigenvalue_range': (evals.min().item(), evals.max().item()),
    }


# ---------------------------------------------------------------------------
# Phase 7: Full-Rank Vocabulary Transfer
# ---------------------------------------------------------------------------

def fullrank_vocab_transfer(
    brain,
    embed_weight: torch.Tensor,
    lm_head_weight: Optional[torch.Tensor],
    strength: float = 1.0,
) -> Dict:
    """
    Transfer vocabulary embeddings without lossy SVD truncation.

    After SOMI has been grown to match the effective rank, hidden_dim
    should be >= the embedding's intrinsic dimension. We transfer as
    many modes as fit, using MP thresholding to skip noise.

    Args:
        brain: SOMICircuitBrain (already grown to target dim)
        embed_weight: [vocab, src_dim] embedding matrix
        lm_head_weight: [vocab, src_dim] or None (often tied to embed)
        strength: blending strength
    """
    diagnostics = {}
    tgt_dim = brain.config.hidden_dim
    device = next(brain.parameters()).device

    if embed_weight is None:
        return {'vocab_absorbed': False, 'reason': 'no_embedding'}

    embed = embed_weight.float()
    vocab_size = embed.shape[0]
    src_dim = embed.shape[1]

    # SVD of embedding matrix
    U, S, Vh = torch.linalg.svd(embed, full_matrices=False)

    # Use MP threshold to determine how many modes carry real information
    mp = marchenko_pastur_threshold(embed)
    K_signal = mp['K_signal']
    K = min(K_signal, tgt_dim, S.shape[0])

    diagnostics['vocab_K_signal'] = K_signal
    diagnostics['vocab_K_used'] = K
    diagnostics['vocab_energy_preserved'] = (
        S[:K].pow(2).sum() / S.pow(2).sum().clamp(min=1e-8)
    ).item()

    # Projection matrix: [K, src_dim]
    P = Vh[:K, :]

    # Project embeddings: [vocab, src_dim] @ [src_dim, K] -> [vocab, K]
    projected = embed @ P.T

    # Install into Y-Decoder: [output_dim, hidden_dim]
    with torch.no_grad():
        dec_w = brain.y_decoder.weight.data
        n_vocab = min(dec_w.shape[0], vocab_size)
        n_hidden = min(K, dec_w.shape[1])

        proj_slice = projected[:n_vocab, :n_hidden].to(device)
        # Scale to match existing decoder distribution
        existing_std = dec_w[:n_vocab, :n_hidden].std().clamp(min=1e-8)
        proj_std = proj_slice.std().clamp(min=1e-8)
        proj_scaled = proj_slice * (existing_std / proj_std)

        dec_w[:n_vocab, :n_hidden] = (
            (1 - strength) * dec_w[:n_vocab, :n_hidden]
            + strength * proj_scaled
        )
        diagnostics['y_decoder_vocab'] = n_vocab
        diagnostics['y_decoder_hidden'] = n_hidden

    # Install into X-Encoder: [hidden_dim, input_dim]
    with torch.no_grad():
        enc_w = brain.x_encoder.weight.data
        n_out = min(K, enc_w.shape[0])
        n_in = min(enc_w.shape[1], K)

        # Weighted projection matrix for encoder
        P_weighted = (S[:K].unsqueeze(1) * P[:K, :])  # [K, src_dim]
        proj_enc = P_weighted[:n_out, :n_in].to(device)

        enc_std = enc_w[:n_out, :n_in].std().clamp(min=1e-8)
        proj_enc_std = proj_enc.std().clamp(min=1e-8)
        proj_enc_scaled = proj_enc * (enc_std / proj_enc_std)

        enc_w[:n_out, :n_in] = (
            (1 - strength) * enc_w[:n_out, :n_in]
            + strength * proj_enc_scaled
        )
        diagnostics['x_encoder_out'] = n_out
        diagnostics['x_encoder_in'] = n_in

    diagnostics['vocab_absorbed'] = True
    return diagnostics


# ---------------------------------------------------------------------------
# Master Pipeline: Lossless Multi-Model Absorption
# ---------------------------------------------------------------------------

def lossless_absorb_all(
    brain,
    model_list: List[Tuple[str, str]],
    device: str = 'cuda',
    grow_to_fit: bool = True,
    target_hidden_dim: Optional[int] = None,
) -> Dict:
    """
    Complete lossless absorption pipeline.

    For each model:
      1. Extract transformer weights
      2. Lossless spectral extraction (MP-filtered)
      3. Procrustes alignment to common coordinate frame
      4. Kalman fusion (optimal sequential weighting)
    Then:
      5. Grow SOMI to fit effective rank
      6. Install fused spectrum at full strength
      7. Full-rank vocabulary transfer

    Args:
        brain: SOMICircuitBrain
        model_list: [(model_id, label), ...]
        device: 'cuda' or 'cpu'
        grow_to_fit: If True, grow brain to match max effective rank
        target_hidden_dim: Override growth target (None = auto from data)

    Returns:
        diagnostics with full absorption report
    """
    from .from_huggingface import extract_transformer_weights
    import shutil
    import os

    diagnostics = {
        'models': [],
        'n_models': len(model_list),
    }

    DEVICE = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ===== PHASE 1-3: Extract, filter, align from each model =====
    all_spectral = []  # Per-model list of per-layer spectral extractions
    all_embeds = []
    reference_evecs = None  # First model's eigenvectors as reference frame
    max_effective_rank = 0
    max_vocab = 0

    for idx, (model_id, label) in enumerate(model_list):
        print(f"\n[Lossless] Model {idx+1}/{len(model_list)}: {model_id} ({label})")

        try:
            # Step 1: Extract raw weights
            weights = extract_transformer_weights(model_id, device=device)
            hidden_dim = weights['hidden_dim']
            n_layers = weights['n_layers']
            vocab_size = weights.get('vocab_size', 0)
            max_vocab = max(max_vocab, vocab_size)

            # Step 2: Lossless spectral extraction per layer
            print(f"  Extracting spectral content ({n_layers} layers, H={hidden_dim})...")
            model_spectral = []
            for li in range(n_layers):
                layer = weights['layers'][li]
                spec = lossless_spectral_extract(layer, hidden_dim, device=DEVICE)
                if spec is not None:
                    model_spectral.append(spec)
                    max_effective_rank = max(max_effective_rank, int(spec['effective_rank']))

            print(f"  Extracted {len(model_spectral)}/{n_layers} layers, "
                  f"max eff_rank={max_effective_rank}")

            # Step 3: Procrustes alignment
            if model_spectral:
                if reference_evecs is None:
                    # First model becomes reference frame
                    reference_evecs = model_spectral[0]['eigenvectors']
                    print(f"  Reference frame set (dim={reference_evecs.shape})")
                else:
                    # Align all layers to reference
                    n_aligned = 0
                    for spec in model_spectral:
                        try:
                            V_aligned, R, err = procrustes_align(
                                reference_evecs, spec['eigenvectors']
                            )
                            spec['eigenvectors'] = V_aligned
                            n_aligned += 1
                        except Exception:
                            pass
                    print(f"  Procrustes aligned {n_aligned}/{len(model_spectral)} layers")

            all_spectral.append(model_spectral)

            # Save embedding for vocab transfer (keep the largest)
            if weights.get('embed_weight') is not None:
                all_embeds.append({
                    'embed_weight': weights['embed_weight'],
                    'lm_head_weight': weights.get('lm_head_weight'),
                    'vocab_size': vocab_size,
                    'hidden_dim': hidden_dim,
                })

            diagnostics['models'].append({
                'model_id': model_id,
                'label': label,
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'n_spectral_layers': len(model_spectral),
                'max_effective_rank': max(
                    [int(s['effective_rank']) for s in model_spectral]
                ) if model_spectral else 0,
            })

            # Clean up
            del weights
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clear HF cache
            cache_dir = os.path.join(
                os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                "hub",
            )
            model_dir = os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
            if os.path.exists(model_dir):
                freed = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fns in os.walk(model_dir) for f in fns
                ) / 1e6
                shutil.rmtree(model_dir, ignore_errors=True)
                print(f"  Freed {freed:.0f} MB cache")

        except Exception as e:
            print(f"  ERROR: {e}")
            diagnostics['models'].append({
                'model_id': model_id, 'label': label, 'error': str(e)
            })

    # ===== PHASE 4: Grow SOMI =====
    if grow_to_fit:
        if target_hidden_dim is None:
            # Use effective rank + 10% headroom, rounded to multiple of 64
            target_hidden_dim = ((int(max_effective_rank * 1.1) + 63) // 64) * 64
            target_hidden_dim = max(target_hidden_dim, brain.config.hidden_dim)

        if target_hidden_dim > brain.config.hidden_dim:
            print(f"\n[Lossless] Growing SOMI: {brain.config.hidden_dim} -> {target_hidden_dim}")
            brain.grow_brain(target_hidden_dim)
            brain = brain.to(DEVICE)
            brain.recalibrate_config()
            diagnostics['grew_to'] = target_hidden_dim

    # Expand output dim for largest vocabulary
    if max_vocab > brain.y_decoder.out_features:
        print(f"[Lossless] Expanding vocab: {brain.y_decoder.out_features} -> {max_vocab}")
        old_state = brain.state_dict()
        from ..config import SOMIBrainConfig
        from ..brain.circuit_brain import SOMICircuitBrain
        new_brain = SOMICircuitBrain(
            brain.config,
            input_dim=brain.x_encoder.in_features,
            output_dim=max_vocab,
        )
        new_state = new_brain.state_dict()
        for k, v in old_state.items():
            if k in new_state:
                if v.shape == new_state[k].shape:
                    new_state[k] = v
                elif v.dim() >= 1 and all(
                    s1 <= s2 for s1, s2 in zip(v.shape, new_state[k].shape)
                ):
                    slices = tuple(slice(0, s) for s in v.shape)
                    new_state[k][slices] = v
        new_brain.load_state_dict(new_state)
        brain = new_brain.to(DEVICE)
        diagnostics['vocab_expanded_to'] = max_vocab

    # ===== PHASE 5: Kalman Fusion =====
    n_parts = len(brain.parts)
    tgt_dim = brain.config.hidden_dim
    n_modes = tgt_dim  # One mode per hidden dimension

    print(f"\n[Lossless] Kalman fusion: {len(all_spectral)} models -> "
          f"{n_parts} Parts (H={tgt_dim})")

    # Create one Kalman fuser per Part
    fusers = [KalmanSpectralFuser(n_modes) for _ in range(n_parts)]
    mass_accum = [torch.zeros(tgt_dim) for _ in range(n_parts)]
    mass_count = [0] * n_parts

    for model_idx, model_spectral in enumerate(all_spectral):
        if not model_spectral:
            continue

        # Distribute layers across Parts
        layers_per_part = max(1, len(model_spectral) // n_parts)
        model_info = diagnostics['models'][model_idx]
        model_weight = model_info.get('hidden_dim', 1000) / 1000.0

        for pid in range(n_parts):
            start = pid * layers_per_part
            end = min(start + layers_per_part, len(model_spectral))
            if start >= len(model_spectral):
                break

            # Average spectral content for this Part's layer group
            group_evals = []
            group_evecs = []
            for li in range(start, end):
                spec = model_spectral[li]
                group_evals.append(spec['eigenvalues'])
                group_evecs.append(spec['eigenvectors'])
                if spec['mass'] is not None:
                    m = spec['mass']
                    if m.shape[0] != tgt_dim:
                        m = torch.nn.functional.interpolate(
                            m.unsqueeze(0).unsqueeze(0),
                            size=tgt_dim, mode='linear', align_corners=False,
                        ).squeeze()
                    mass_accum[pid] += m.cpu()
                    mass_count[pid] += 1

            if not group_evals:
                continue

            # Average eigenvalues across layers in this group
            min_k_eval = min(e.shape[0] for e in group_evals)
            min_k_evec = min(v.shape[1] for v in group_evecs)
            min_k = min(min_k_eval, min_k_evec)
            avg_evals = torch.stack([e[-min_k:] for e in group_evals]).mean(0)

            # Average eigenvectors
            min_dim = min(v.shape[0] for v in group_evecs)
            avg_evecs = torch.stack([v[:min_dim, -min_k:] for v in group_evecs]).mean(0)
            # Re-normalize columns
            norms = avg_evecs.norm(dim=0, keepdim=True).clamp(min=1e-8)
            avg_evecs = avg_evecs / norms

            # Kalman update
            noise = 1.0 / model_weight  # larger models = lower noise
            fusers[pid].absorb(
                avg_evals.cpu(), avg_evecs.cpu(),
                measurement_noise=noise,
                model_weight=model_weight,
            )

    # ===== PHASE 6: Install Fused Spectrum =====
    print(f"\n[Lossless] Installing fused spectrum into {n_parts} Parts...")

    for pid, (part_name, part) in enumerate(brain.parts.items()):
        if pid >= len(fusers) or fusers[pid].n_absorbed == 0:
            continue

        fused_evals, fused_evecs = fusers[pid].get_fused_spectrum()

        diag = install_spectrum_into_part(
            part, fused_evals, fused_evecs, strength=1.0
        )
        print(f"  Part {part_name}: installed {diag['K_installed']} modes, "
              f"W_change={diag['W_change']:.4f}")

        # Install mass
        if mass_count[pid] > 0:
            avg_mass = mass_accum[pid] / mass_count[pid]
            with torch.no_grad():
                m = avg_mass[:tgt_dim].to(part.mass.device).clamp(0.1, 10.0)
                part.mass.copy_(m)

        diagnostics[f'part_{pid}_installed'] = diag
        diagnostics[f'part_{pid}_kalman_history'] = fusers[pid].history

    # ===== PHASE 7: Full-Rank Vocabulary Transfer =====
    if all_embeds:
        # Use the largest embedding (most vocab coverage)
        best_embed = max(all_embeds, key=lambda e: e['vocab_size'])
        print(f"\n[Lossless] Vocabulary transfer: {best_embed['vocab_size']} tokens, "
              f"H={best_embed['hidden_dim']}")

        vocab_diag = fullrank_vocab_transfer(
            brain,
            best_embed['embed_weight'].to(DEVICE),
            best_embed.get('lm_head_weight', best_embed['embed_weight']).to(DEVICE),
            strength=1.0,
        )
        diagnostics['vocab'] = vocab_diag
        print(f"  Energy preserved: {vocab_diag.get('vocab_energy_preserved', 0):.1%}, "
              f"K_used={vocab_diag.get('vocab_K_used', 0)}")

    # ===== FINAL: Recalibrate =====
    brain.recalibrate_config()
    for part in brain.parts.values():
        part._update_eigen()

    diagnostics['final_hidden_dim'] = brain.config.hidden_dim
    diagnostics['final_n_parts'] = len(brain.parts)
    diagnostics['final_params'] = sum(p.numel() for p in brain.parameters())
    diagnostics['max_effective_rank'] = max_effective_rank
    diagnostics['n_models_absorbed'] = sum(
        1 for m in diagnostics['models'] if 'error' not in m
    )

    print(f"\n[Lossless] COMPLETE: {diagnostics['n_models_absorbed']}/"
          f"{len(model_list)} models absorbed into "
          f"H={brain.config.hidden_dim}, P={len(brain.parts)}, "
          f"params={diagnostics['final_params']:,}")

    return diagnostics
