"""
SOMI 4.0 Mass-Guided Quantization
====================================

Heavy features (high mass = important, slow-changing) need high precision.
Light features (low mass = less important, fast-changing) can use lower precision.

This is the opposite of most quantization: instead of uniform precision,
SOMI uses its physics (mass) to decide WHERE precision matters.
"""

import torch
from typing import Dict, Tuple


def mass_guided_quantization(
    W: torch.Tensor,
    mass: torch.Tensor,
    high_precision_threshold: float = 0.7,
    low_precision_bits: int = 8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Apply mass-guided mixed precision to W.

    Features with mass > threshold keep full precision.
    Features with mass < threshold are quantized to lower precision.

    Args:
        W: [hidden, hidden] connectivity
        mass: [hidden] per-feature mass
        high_precision_threshold: Mass percentile for full precision
        low_precision_bits: Bits for low-precision features

    Returns:
        W_compressed: Quantized connectivity
        diagnostics: Compression metrics
    """
    H = W.shape[0]

    # Determine threshold
    threshold = torch.quantile(mass, high_precision_threshold).item()
    high_mask = mass >= threshold
    low_mask = mass < threshold

    n_high = high_mask.sum().item()
    n_low = low_mask.sum().item()

    # Quantize low-mass rows/cols
    W_compressed = W.clone()
    if n_low > 0 and low_precision_bits < 32:
        # Simple quantization: round to nearest representable value
        scale = W[low_mask, :].abs().max() / (2 ** (low_precision_bits - 1))
        if scale > 0:
            W_compressed[low_mask, :] = (
                (W[low_mask, :] / scale).round() * scale
            )
            W_compressed[:, low_mask] = (
                (W[:, low_mask] / scale).round() * scale
            )

    # Compute compression ratio
    original_bits = H * H * 32  # All fp32
    compressed_bits = (n_high * H + n_low * H) * low_precision_bits + n_high * H * 32
    compression_ratio = original_bits / max(compressed_bits, 1)

    # Quality: how much did W change?
    error = (W - W_compressed).abs().mean().item()

    diagnostics = {
        'compression_n_high_precision': n_high,
        'compression_n_low_precision': n_low,
        'compression_ratio': compression_ratio,
        'compression_quantization_error': error,
        'compression_mass_threshold': threshold,
    }

    return W_compressed, diagnostics
