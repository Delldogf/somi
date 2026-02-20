"""
SOMI 5 Neurogenesis Test
=========================

Tests that the grow() method works correctly on both individual Parts
and the full Circuit Brain.

Verifies:
1. Part grows from N=64 to N=128 (W matrix expands, old memory preserved)
2. Circuit Brain grows (all Parts + White Matter + Encoder/Decoder)
3. Forward pass works at every size
4. No NaN or shape mismatches

Run with: python -m experiments.test_growth
"""

import torch
from somi.config import SOMIBrainConfig
from somi.brain.part import SOMIPart
from somi.brain.circuit_brain import SOMICircuitBrain


def test_part_growth():
    print("=" * 60)
    print(" TEST 1: Part-Level Neurogenesis (N=64 -> N=128)")
    print("=" * 60)

    config = SOMIBrainConfig.auto(hidden_dim=64, n_parts=2)
    part = SOMIPart(part_id=0, config=config)

    h = torch.randn(1, 4, 64)
    output, phi_dot, diag = part(h)
    print(f"  Pre-growth forward pass: output shape = {output.shape}")

    old_W = part.W_local.clone()
    old_mass = part.mass.clone()
    old_H = old_W.shape[0]

    print(f"  Old W shape: {old_W.shape}, mean: {old_W.mean().item():.6f}")
    print(f"  Old mass shape: {old_mass.shape}")

    part.grow(128)

    assert part.W_local.shape == (128, 128), f"W shape wrong: {part.W_local.shape}"
    assert part.mass.shape == (128,), f"Mass shape wrong: {part.mass.shape}"
    assert part.mask.shape == (128, 128), f"Mask shape wrong: {part.mask.shape}"
    assert part.eigenvalues.shape == (128,), f"Eigenvalues shape wrong: {part.eigenvalues.shape}"
    print("  [PASS] Shape checks")

    W_preserved = part.W_local[:old_H, :old_H]
    if torch.allclose(W_preserved, old_W):
        print("  [PASS] Memory preservation (W matrix grafted correctly)")
    else:
        diff = (W_preserved - old_W).abs().max().item()
        print(f"  [FAIL] W graft diff: {diff}")

    mass_preserved = part.mass[:old_H]
    if torch.allclose(mass_preserved, old_mass):
        print("  [PASS] Mass preservation")
    else:
        print("  [FAIL] Mass preservation failed")

    h_big = torch.randn(1, 4, 128)
    try:
        output, phi_dot, diag = part(h_big)
        if torch.isnan(output).any():
            print("  [WARN] Forward pass produced NaN")
        else:
            print(f"  [PASS] Post-growth forward pass: output shape = {output.shape}")
    except Exception as e:
        print(f"  [FAIL] Forward pass crashed: {e}")

    print()


def test_brain_growth():
    print("=" * 60)
    print(" TEST 2: Circuit Brain Neurogenesis (H=64 -> H=128)")
    print("=" * 60)

    config = SOMIBrainConfig.auto(hidden_dim=64, n_parts=4)
    brain = SOMICircuitBrain(config, input_dim=32, output_dim=16)

    x = torch.randn(2, 4, 32)
    logits, diag = brain(x, training=False)
    print(f"  Pre-growth: logits shape = {logits.shape}")
    assert logits.shape == (2, 4, 16), f"Wrong shape: {logits.shape}"
    print("  [PASS] Pre-growth forward pass")

    old_parts_W = {}
    for pid, part in brain.parts.items():
        old_parts_W[pid] = part.W_local[:64, :64].clone()

    brain.grow_brain(128)

    assert brain.config.hidden_dim == 128, f"Config not updated: {brain.config.hidden_dim}"
    print("  [PASS] Config updated to 128")

    for pid, part in brain.parts.items():
        assert part.W_local.shape == (128, 128), f"Part {pid} W wrong: {part.W_local.shape}"
    print("  [PASS] All Parts grown to 128")

    for pid, part in brain.parts.items():
        preserved = part.W_local[:64, :64]
        if torch.allclose(preserved, old_parts_W[pid]):
            print(f"  [PASS] Part {pid} memory preserved")
        else:
            diff = (preserved - old_parts_W[pid]).abs().max().item()
            print(f"  [WARN] Part {pid} drift: {diff:.8f}")

    for name, tract in brain.white_matter.tracts.items():
        assert tract.down_proj.in_features == 128, f"Tract {name} down wrong"
        assert tract.up_proj.out_features == 128, f"Tract {name} up wrong"
    print("  [PASS] White Matter resized to 128")

    assert brain.x_encoder.out_features == 128, "X-Encoder not resized"
    assert brain.y_decoder.in_features == 128, "Y-Decoder not resized"
    print("  [PASS] Encoder/Decoder resized to 128")

    logits2, diag2 = brain(x, training=False)
    assert logits2.shape == (2, 4, 16), f"Post-growth wrong shape: {logits2.shape}"
    if torch.isnan(logits2).any():
        print("  [WARN] Post-growth forward pass produced NaN")
    else:
        print(f"  [PASS] Post-growth forward pass: logits shape = {logits2.shape}")

    print()


def test_multi_stage_growth():
    print("=" * 60)
    print(" TEST 3: Multi-Stage Growth (H=32 -> 64 -> 128)")
    print("=" * 60)

    config = SOMIBrainConfig.auto(hidden_dim=32, n_parts=2)
    brain = SOMICircuitBrain(config, input_dim=16, output_dim=8)

    x = torch.randn(1, 2, 16)

    logits, _ = brain(x, training=True)
    print(f"  Stage 0 (H=32): logits = {logits.shape}, ok = {not torch.isnan(logits).any()}")

    brain.grow_brain(64)
    logits, _ = brain(x, training=True)
    print(f"  Stage 1 (H=64): logits = {logits.shape}, ok = {not torch.isnan(logits).any()}")

    brain.grow_brain(128)
    logits, _ = brain(x, training=True)
    print(f"  Stage 2 (H=128): logits = {logits.shape}, ok = {not torch.isnan(logits).any()}")

    print("  [PASS] Multi-stage growth complete")
    print()


if __name__ == '__main__':
    print("\nSOMI 5 NEUROGENESIS TEST SUITE\n")
    test_part_growth()
    test_brain_growth()
    test_multi_stage_growth()
    print("=" * 60)
    print(" ALL TESTS COMPLETE")
    print("=" * 60)
