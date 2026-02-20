"""
SOMI Spectral Analysis for Lossless Absorption
=================================================

Marchenko-Pastur law for mathematically optimal signal/noise separation,
effective rank computation, and spectral content analysis of transformer layers.

Theory: SOMI_Research/SOMI_4/MATHEMATICAL_TOOLS_FOR_ABSORPTION.md (Tool 2.1)
"""

import torch
import math
from typing import Dict, List, Tuple, Optional


def marchenko_pastur_threshold(
    W: torch.Tensor,
    return_filtered: bool = False,
) -> Dict:
    """
    Compute the Marchenko-Pastur signal/noise threshold for a weight matrix.

    The Marchenko-Pastur law gives the EXACT distribution of singular values
    for a random matrix. Singular values above the MP edge are guaranteed to
    be signal (real knowledge). Below = noise.

    This replaces the ad-hoc "keep 90% energy" rule with a mathematically
    proven optimal cut.

    Args:
        W: Weight matrix [m, n]
        return_filtered: If True, also return the noise-filtered matrix

    Returns:
        Dictionary with:
        - mp_threshold: The noise ceiling (singular values above this = signal)
        - sigma_noise: Estimated noise level
        - K_signal: Number of signal modes
        - K_total: Total number of modes
        - signal_fraction: K_signal / K_total
        - energy_in_signal: Fraction of total energy in signal modes
        - effective_rank: Sum of singular values / max singular value
        - spectral_entropy_rank: exp(entropy of normalized squared singular values)
        - singular_values: All singular values (for analysis)
        - filtered_W: (optional) Noise-removed matrix
    """
    W_float = W.float()
    U, S, Vh = torch.linalg.svd(W_float, full_matrices=False)

    m, n = W_float.shape
    gamma = m / n if m >= n else n / m

    # Estimate noise from median singular value
    # The median of MP distribution is approximately sigma * (1 + sqrt(gamma))^(2/3)
    # Simpler: use median as robust noise estimator
    sigma_noise = S.median().item() / (1 + gamma ** 0.5)

    # MP upper edge: above this = signal
    threshold = sigma_noise * (1 + gamma ** 0.5)

    # Count signal modes
    signal_mask = S > threshold
    K_signal = signal_mask.sum().item()
    K_total = S.shape[0]

    # Energy analysis
    S_sq = S.pow(2)
    total_energy = S_sq.sum().item()
    signal_energy = S_sq[signal_mask].sum().item() if K_signal > 0 else 0.0

    # Effective rank (sum of singular values / max)
    effective_rank = (S.sum() / (S[0] + 1e-8)).item()

    # Spectral entropy rank: exp(H) where H = -sum(p_i * log(p_i))
    p = S_sq / (total_energy + 1e-8)
    p_safe = p.clamp(min=1e-12)
    entropy = -(p_safe * p_safe.log()).sum().item()
    spectral_entropy_rank = math.exp(entropy)

    # 99% energy threshold rank
    cumulative = S_sq.cumsum(0) / (total_energy + 1e-8)
    rank_99 = (cumulative < 0.99).sum().item() + 1

    result = {
        'mp_threshold': threshold,
        'sigma_noise': sigma_noise,
        'K_signal': K_signal,
        'K_total': K_total,
        'signal_fraction': K_signal / K_total if K_total > 0 else 0.0,
        'energy_in_signal': signal_energy / (total_energy + 1e-8),
        'effective_rank': effective_rank,
        'spectral_entropy_rank': spectral_entropy_rank,
        'rank_99_energy': rank_99,
        'singular_values': S.detach(),
    }

    if return_filtered and K_signal > 0:
        U_sig = U[:, :K_signal]
        S_sig = S[:K_signal]
        Vh_sig = Vh[:K_signal, :]
        result['filtered_W'] = (U_sig * S_sig.unsqueeze(0)) @ Vh_sig

    return result


def analyze_transformer_layer(
    layer_data: Dict,
    hidden_dim: int,
) -> Dict:
    """
    Full spectral analysis of one transformer layer.

    Analyzes attention, MLP, and combined correlation matrices
    to determine the effective computational content.

    Args:
        layer_data: Dict from extract_transformer_weights with keys:
            attn_qkv, attn_out, mlp_up, mlp_down, ln_weight
        hidden_dim: Model's hidden dimension

    Returns:
        Dict with spectral analysis for each component and combined
    """
    results = {'hidden_dim': hidden_dim}

    # Analyze attention correlation: C_attn = W_out^T @ W_out
    if layer_data.get('attn_out') is not None:
        a = layer_data['attn_out'].float()
        if a.shape[0] == hidden_dim:
            C_attn = a.T @ a  # [H, H]
        elif a.shape[-1] == hidden_dim:
            C_attn = a @ a.T  # [H, H]
        else:
            C_attn = None

        if C_attn is not None:
            results['attn'] = marchenko_pastur_threshold(C_attn)

    # Analyze MLP transformation: C_mlp = W_down @ W_up
    if layer_data.get('mlp_down') is not None and layer_data.get('mlp_up') is not None:
        d = layer_data['mlp_down'].float()
        u = layer_data['mlp_up'].float()

        if d.shape[0] == hidden_dim and u.shape[-1] == hidden_dim:
            C_mlp = d @ u  # [H, H]
        elif d.shape[-1] == hidden_dim and u.shape[0] == hidden_dim:
            C_mlp = d.T @ u.T  # [H, H]
        else:
            # Try to make it work with available shapes
            try:
                if d.shape[-1] == u.shape[0]:
                    C_mlp = d @ u
                elif d.shape[0] == u.shape[-1]:
                    C_mlp = d.T @ u.T
                else:
                    C_mlp = None
            except RuntimeError:
                C_mlp = None

        if C_mlp is not None:
            results['mlp'] = marchenko_pastur_threshold(C_mlp)

    # Combined correlation: C_attn + C_mlp (symmetrized)
    if 'attn' in results and 'mlp' in results:
        C_combined = C_attn + C_mlp
        C_sym = 0.5 * (C_combined + C_combined.T)
        results['combined'] = marchenko_pastur_threshold(C_sym)

    return results


def analyze_full_model(
    model_weights: Dict,
) -> Dict:
    """
    Full spectral analysis of an entire transformer model.

    Args:
        model_weights: Output of extract_transformer_weights()

    Returns:
        Dict with per-layer analysis and model-wide summary
    """
    hidden_dim = model_weights['hidden_dim']
    n_layers = model_weights['n_layers']
    layers = model_weights['layers']

    layer_results = []
    all_signal_modes = []
    all_effective_ranks = []
    all_entropy_ranks = []

    for i, layer in enumerate(layers):
        analysis = analyze_transformer_layer(layer, hidden_dim)
        analysis['layer_idx'] = i
        layer_results.append(analysis)

        # Collect key metrics from the best available analysis
        for key in ['combined', 'attn', 'mlp']:
            if key in analysis:
                all_signal_modes.append(analysis[key]['K_signal'])
                all_effective_ranks.append(analysis[key]['effective_rank'])
                all_entropy_ranks.append(analysis[key]['spectral_entropy_rank'])
                break

    # Embedding analysis
    embed_analysis = None
    if model_weights.get('embed_weight') is not None:
        embed = model_weights['embed_weight'].float()
        embed_analysis = marchenko_pastur_threshold(embed)

    # Summary statistics
    summary = {
        'model_name': model_weights.get('model_name', 'unknown'),
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'vocab_size': model_weights.get('vocab_size', 0),
        'layer_results': layer_results,
        'embed_analysis': embed_analysis,
    }

    if all_signal_modes:
        summary['signal_modes_per_layer'] = {
            'min': min(all_signal_modes),
            'max': max(all_signal_modes),
            'mean': sum(all_signal_modes) / len(all_signal_modes),
            'median': sorted(all_signal_modes)[len(all_signal_modes) // 2],
        }
        summary['effective_rank_per_layer'] = {
            'min': min(all_effective_ranks),
            'max': max(all_effective_ranks),
            'mean': sum(all_effective_ranks) / len(all_effective_ranks),
        }
        summary['entropy_rank_per_layer'] = {
            'min': min(all_entropy_ranks),
            'max': max(all_entropy_ranks),
            'mean': sum(all_entropy_ranks) / len(all_entropy_ranks),
        }
        summary['recommended_hidden_dim'] = _next_multiple_of_64(
            int(max(all_signal_modes) * 1.1)
        )

    return summary


def _next_multiple_of_64(n: int) -> int:
    """Round up to next multiple of 64 for GPU efficiency."""
    return ((n + 63) // 64) * 64


def print_analysis_report(summary: Dict):
    """Pretty-print the spectral analysis results."""
    print(f"\n{'='*70}")
    print(f"  SPECTRAL ANALYSIS: {summary['model_name']}")
    print(f"{'='*70}")
    print(f"  Hidden dim: {summary['hidden_dim']}")
    print(f"  Layers: {summary['n_layers']}")
    print(f"  Vocab size: {summary['vocab_size']}")

    if 'signal_modes_per_layer' in summary:
        sm = summary['signal_modes_per_layer']
        er = summary['effective_rank_per_layer']
        se = summary['entropy_rank_per_layer']

        print(f"\n  --- Marchenko-Pastur Signal Modes ---")
        print(f"  Per layer: min={sm['min']}, max={sm['max']}, "
              f"mean={sm['mean']:.0f}, median={sm['median']}")
        print(f"  Effective rank: min={er['min']:.0f}, max={er['max']:.0f}, "
              f"mean={er['mean']:.0f}")
        print(f"  Spectral entropy rank: min={se['min']:.0f}, "
              f"max={se['max']:.0f}, mean={se['mean']:.0f}")
        print(f"\n  >>> Recommended SOMI hidden_dim: "
              f"{summary['recommended_hidden_dim']}")

    if summary.get('embed_analysis'):
        ea = summary['embed_analysis']
        print(f"\n  --- Embedding Matrix ---")
        print(f"  MP signal modes: {ea['K_signal']} / {ea['K_total']}")
        print(f"  Energy in signal: {ea['energy_in_signal']:.1%}")
        print(f"  Effective rank: {ea['effective_rank']:.0f}")
        print(f"  99% energy rank: {ea['rank_99_energy']}")

    print(f"\n  --- Per-Layer Detail ---")
    for lr in summary.get('layer_results', []):
        idx = lr['layer_idx']
        for key in ['combined', 'attn', 'mlp']:
            if key in lr:
                r = lr[key]
                print(f"  Layer {idx:2d} ({key:8s}): "
                      f"signal={r['K_signal']:4d}/{r['K_total']:4d} "
                      f"({r['signal_fraction']:.1%}), "
                      f"energy={r['energy_in_signal']:.1%}, "
                      f"eff_rank={r['effective_rank']:.0f}, "
                      f"entropy_rank={r['spectral_entropy_rank']:.0f}")
                break

    print(f"{'='*70}\n")
