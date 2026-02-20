"""
Absorb knowledge from a HuggingFace LLM into SOMI.

Usage:
    python -m experiments.absorb_from_llm

This will:
  1. Download a small open-source LLM (Qwen2.5-0.5B by default)
  2. Extract its weight matrices (attention patterns, MLP, LayerNorm)
  3. Create a SOMI brain and absorb the knowledge
  4. Run integrity checks
  5. Compare before/after fingerprints

Requirements:
    pip install transformers
"""

import torch
from somi.config import SOMIBrainConfig
from somi.brain.circuit_brain import SOMICircuitBrain
from somi.absorption.from_huggingface import (
    extract_transformer_weights,
    absorb_weights_into_brain,
)
from somi.absorption.fingerprint import compute_fingerprint
from somi.absorption.integrity import check_integrity


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
SOMI_HIDDEN = 128
SOMI_PARTS = 4
MAX_LAYERS = 8  # Only use first 8 layers (saves memory)
STRENGTH = 0.7
METHOD = 'direct'  # 'direct' or 'spectral'


def main():
    print("=" * 60)
    print(f"  Absorbing from: {MODEL_NAME}")
    print(f"  SOMI: H={SOMI_HIDDEN}, P={SOMI_PARTS}")
    print(f"  Method: {METHOD}, Strength: {STRENGTH}")
    print("=" * 60)

    # Step 0: Extract weights first so we know the vocab size
    print(f"\n--- Extracting weights from {MODEL_NAME} ---")
    weights = extract_transformer_weights(MODEL_NAME, device='cpu', max_layers=MAX_LAYERS)
    vocab_size = weights['vocab_size']
    print(f"  Source vocab size: {vocab_size}")

    # Step 1: Create SOMI brain with MATCHING vocab size
    config = SOMIBrainConfig.auto(SOMI_HIDDEN, SOMI_PARTS)
    brain = SOMICircuitBrain(config, input_dim=SOMI_HIDDEN, output_dim=vocab_size)

    # Step 2: Fingerprint BEFORE absorption
    print("\n--- Fingerprint BEFORE absorption ---")
    fp_before = compute_fingerprint(brain)
    for k, v in sorted(fp_before.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")

    # Step 3: Absorb (weights already extracted above)
    print(f"\n--- Absorbing into SOMI ---")
    diag = absorb_weights_into_brain(brain, weights, strength=STRENGTH, method=METHOD)

    print("\nAbsorption diagnostics:")
    for k, v in sorted(diag.items()):
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")

    # Step 5: Integrity check
    print("\n--- Integrity check ---")
    integrity = check_integrity(brain, verbose=True)
    print(f"  Overall: {'HEALTHY' if integrity['overall_healthy'] else 'ISSUES'}")

    # Step 6: Fingerprint AFTER absorption
    print("\n--- Fingerprint AFTER absorption ---")
    fp_after = compute_fingerprint(brain)
    for k, v in sorted(fp_after.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")

    # Step 7: Compare
    print("\n--- Changes ---")
    for k in sorted(fp_before):
        if k in fp_after and isinstance(fp_before[k], float):
            change = fp_after[k] - fp_before[k]
            direction = "+" if change > 0 else ""
            print(f"  {k}: {direction}{change:.6f}")

    # Step 8: Test forward pass
    print("\n--- Forward pass test ---")
    x = torch.randn(2, 8, SOMI_HIDDEN)
    logits, brain_diag = brain(x, training=False)
    methods = [v for k, v in brain_diag.items() if 'method' in k]
    print(f"  Output: {logits.shape}")
    print(f"  Settle methods: {methods}")
    print(f"  Diagnostics: {len(brain_diag)} metrics")

    print("\n" + "=" * 60)
    print("  ABSORPTION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
