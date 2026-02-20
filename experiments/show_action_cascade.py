"""
Show Action-Derived Parameter Cascade
========================================

Demonstrates that ALL SOMI parameters are derived from the single action
S[phi, W] given only hidden_dim and n_parts.

Shows:
  1. Full derivation for each standard scale (S, M, L, XL)
  2. Side-by-side comparison of key parameters across scales
  3. Quick functional test: build a brain with auto() and forward-pass

Run with: python -m experiments.show_action_cascade
"""

from somi.physics.action_derived import (
    derive_all_from_action,
    print_derivation,
    compare_scales,
)
from somi.config import SOMIBrainConfig


def main():
    # ---- 1. Full derivation for Circuit-S ----
    print("\n" + "=" * 70)
    print("  PART 1: Full derivation chain (Circuit-S)")
    print("=" * 70)
    print_derivation(128, 4)

    # ---- 2. Side-by-side across all scales ----
    print("\n" + "=" * 70)
    print("  PART 2: How parameters scale across sizes")
    print("=" * 70 + "\n")
    compare_scales()

    # ---- 3. Functional test ----
    print("\n" + "=" * 70)
    print("  PART 3: Functional test — build brain with auto() and forward")
    print("=" * 70 + "\n")

    import torch
    from somi.brain.circuit_brain import SOMICircuitBrain

    for name, N, P in [('Circuit-S', 128, 4), ('Circuit-M', 256, 8)]:
        config = SOMIBrainConfig.auto(hidden_dim=N, n_parts=P)
        brain = SOMICircuitBrain(config, input_dim=64, output_dim=100)

        x = torch.randn(2, 8, 64)
        logits, diag = brain(x, training=True)

        n_params = sum(p.numel() for p in brain.parameters())
        print(f"  {name}: hidden={N}, parts={P}")
        print(f"    Params: {n_params:,}")
        print(f"    alpha_1={config.alpha_1:.4f}  dt={config.dt:.4f}  "
              f"zeta={config.target_zeta:.4f}  timescale={config.timescale_ratio:.2f}")
        print(f"    Output: {list(logits.shape)}")
        print(f"    Diagnostics: {len(diag)} metrics")
        print()

    print("  All parameters derived from the action. Zero free parameters.")
    print()


if __name__ == '__main__':
    main()
