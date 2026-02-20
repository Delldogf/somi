"""
Lossless Spectral Absorption Experiment
==========================================

Runs the complete 7-phase lossless absorption pipeline:
1. Extract spectral content from 8 open-source LLMs
2. Marchenko-Pastur noise filtering
3. Procrustes cross-model alignment
4. Grow SOMI to match effective rank
5. Kalman-optimal multi-model fusion
6. Full-strength spectral installation
7. Full-rank vocabulary transfer

Then validates with forward pass and integrity checks.
"""

import sys
import os
import json
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from somi.config import SOMIBrainConfig
from somi.brain.circuit_brain import SOMICircuitBrain
from somi.absorption.lossless import lossless_absorb_all
from somi.absorption.integrity import check_integrity
from somi.absorption.fingerprint import compute_fingerprint
from somi.checkpoint import save_checkpoint

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "0.5B"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B"),
    ("Qwen/Qwen2.5-1.5B", "1.5B"),
    ("stabilityai/stablelm-2-1_6b", "1.6B"),
    ("EleutherAI/pythia-2.8b", "2.8B"),
    ("Qwen/Qwen2.5-3B", "3B"),
    ("Qwen/Qwen2.5-7B", "7B"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INITIAL_HIDDEN = 256
INITIAL_PARTS = 4
INITIAL_VOCAB = 32000
CHECKPOINT_DIR = "/workspace/somi_lossless" if os.path.exists("/workspace") else "somi_lossless"
RESULTS_DIR = "/workspace" if os.path.exists("/workspace") else "."


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  LOSSLESS SPECTRAL ABSORPTION EXPERIMENT")
    print(f"{'='*70}")
    print(f"  Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    print(f"  Models: {len(MODELS)}")
    print(f"  Initial SOMI: H={INITIAL_HIDDEN}, P={INITIAL_PARTS}")
    print(f"  Will grow to match effective rank")
    print(f"{'='*70}\n")

    # Create initial SOMI brain (small — will be grown)
    config = SOMIBrainConfig.auto(INITIAL_HIDDEN, INITIAL_PARTS)
    brain = SOMICircuitBrain(
        config,
        input_dim=INITIAL_HIDDEN,
        output_dim=INITIAL_VOCAB,
    ).to(DEVICE)

    initial_params = sum(p.numel() for p in brain.parameters())
    print(f"  Initial brain: {initial_params:,} params")

    # Pre-absorption fingerprint
    fp_before = compute_fingerprint(brain)

    # Run the full lossless pipeline
    t_start = time.time()

    diagnostics = lossless_absorb_all(
        brain,
        model_list=MODELS,
        device=DEVICE,
        grow_to_fit=True,
        target_hidden_dim=None,  # auto from MP analysis
    )

    t_total = time.time() - t_start

    # Post-absorption fingerprint
    fp_after = compute_fingerprint(brain)

    # Integrity check
    print(f"\n{'='*70}")
    print(f"  VALIDATION")
    print(f"{'='*70}")

    integrity = check_integrity(brain, verbose=True)
    diagnostics['integrity'] = integrity

    # Forward pass test
    print(f"\n  Forward pass test...")
    try:
        with torch.no_grad():
            x = torch.randn(2, 16, brain.config.hidden_dim, device=DEVICE)
            output = brain(x)
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output
            print(f"  Output shape: {out_tensor.shape}")
            print(f"  Output range: [{out_tensor.min():.3f}, {out_tensor.max():.3f}]")
            diagnostics['forward_pass'] = {
                'output_shape': list(out_tensor.shape),
                'output_min': out_tensor.min().item(),
                'output_max': out_tensor.max().item(),
                'success': True,
            }
    except Exception as e:
        print(f"  Forward pass FAILED: {e}")
        diagnostics['forward_pass'] = {'success': False, 'error': str(e)}

    # Fingerprint comparison
    print(f"\n  Fingerprint comparison:")
    for key in sorted(set(list(fp_before.keys()) + list(fp_after.keys()))):
        before = fp_before.get(key, 0)
        after = fp_after.get(key, 0)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            if abs(after - before) > 0.001:
                print(f"    {key}: {before:.4f} -> {after:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "somi_lossless_absorbed.pt")
    save_checkpoint(brain, ckpt_path, extra_metadata={
        'absorption_diagnostics': str(diagnostics.get('n_models_absorbed', 0)),
        'hidden_dim': brain.config.hidden_dim,
    })
    ckpt_size = os.path.getsize(ckpt_path) / 1e6
    print(f"\n  Checkpoint: {ckpt_path} ({ckpt_size:.1f} MB)")

    # Save results
    results_path = os.path.join(RESULTS_DIR, "lossless_absorb_results.json")
    clean_diag = {}
    for k, v in diagnostics.items():
        try:
            json.dumps(v, default=str)
            clean_diag[k] = v
        except (TypeError, ValueError):
            clean_diag[k] = str(v)

    with open(results_path, "w") as f:
        json.dump(clean_diag, f, indent=2, default=str)
    print(f"  Results: {results_path}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  COMPLETE")
    print(f"  Models absorbed: {diagnostics.get('n_models_absorbed', 0)}/{len(MODELS)}")
    print(f"  Final hidden_dim: {diagnostics.get('final_hidden_dim', '?')}")
    print(f"  Final params: {diagnostics.get('final_params', '?'):,}")
    print(f"  Max effective rank: {diagnostics.get('max_effective_rank', '?')}")
    print(f"  Integrity: {'HEALTHY' if integrity.get('overall_healthy') else 'ISSUES'}")
    print(f"  Total time: {t_total:.0f}s ({t_total/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
