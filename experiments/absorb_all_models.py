"""
Absorb All Models: Multi-Model Absorption into One SOMI Brain
===============================================================

Downloads and absorbs open-source LLMs one at a time into a single
growing SOMI brain. Each model is downloaded, absorbed, then its
cache is deleted to save disk space.

Usage (on RunPod or any GPU machine):
    export HF_HOME=/workspace/.cache/huggingface
    python -m experiments.absorb_all_models

The script:
  For each model: Download -> Extract -> Absorb -> Delete cache
  Final: Run diagnostics and save checkpoint
"""

import os
import sys
import time
import json
import shutil
import torch
from pathlib import Path
from datetime import datetime

from somi.config import SOMIBrainConfig
from somi.brain.circuit_brain import SOMICircuitBrain
from somi.absorption.from_huggingface import (
    extract_transformer_weights,
    absorb_weights_into_brain,
)
from somi.absorption.fingerprint import compute_fingerprint
from somi.absorption.integrity import check_integrity
from somi.checkpoint import save_checkpoint

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ============================================================
# Configuration — all non-gated models (no license click needed)
# ============================================================

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "0.5B"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B"),
    ("Qwen/Qwen2.5-1.5B", "1.5B"),
    ("stabilityai/stablelm-2-1_6b", "1.6B"),
    ("microsoft/phi-2", "2.7B"),
    ("EleutherAI/pythia-2.8b", "2.8B"),
    ("Qwen/Qwen2.5-3B", "3B"),
    ("Qwen/Qwen2.5-7B", "7B"),
]

SOMI_HIDDEN = 256
SOMI_PARTS = 4
STRENGTH = 0.7
METHOD = 'direct'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_DIR = Path("/workspace/somi_checkpoints")
FINAL_DIR = Path("/workspace/somi_final")
LOG_FILE = Path("/workspace/absorb_results.json")


# ============================================================
# Download + Absorb (one model at a time to save disk)
# ============================================================

def download_model(model_id):
    """Download a single model. Returns True on success."""
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(model_id, ignore_patterns=["*.gguf", "*.bin.index.json"])
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


def clear_model_cache(model_id):
    """Delete cached model files to free disk space."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = Path(hf_home) / "hub"
    safe_name = "models--" + model_id.replace("/", "--")
    cache_path = hub_dir / safe_name
    if cache_path.exists():
        size_mb = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file()) / 1e6
        shutil.rmtree(cache_path, ignore_errors=True)
        print(f"    Freed {size_mb:.0f} MB from cache")


def absorb_all():
    """Download, absorb, and delete each model one at a time."""
    print("\n" + "=" * 70)
    print("  ABSORBING MODELS INTO SOMI (one at a time)")
    print("=" * 70)
    print(f"  Models: {len(MODELS)}")
    print(f"  SOMI config: H={SOMI_HIDDEN}, P={SOMI_PARTS}, device={DEVICE}")
    print(f"  Method: {METHOD}, Strength: {STRENGTH}")

    brain = None
    config = SOMIBrainConfig.auto(SOMI_HIDDEN, SOMI_PARTS)
    all_results = {}
    absorbed_count = 0

    if HAS_WANDB:
        wandb.init(
            project="somi-absorb-all",
            name=f"absorb-{len(MODELS)}-models-{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "somi_hidden": SOMI_HIDDEN,
                "somi_parts": SOMI_PARTS,
                "strength": STRENGTH,
                "method": METHOD,
                "n_models": len(MODELS),
                "models": [m for m, _ in MODELS],
                "device": DEVICE,
            },
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for i, (model_id, size) in enumerate(MODELS, start=1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(MODELS)}] {model_id} ({size})")
        print(f"{'='*60}")

        # Step 1: Download
        print(f"  Downloading...")
        t0 = time.time()
        ok = download_model(model_id)
        print(f"  Download: {'OK' if ok else 'FAILED'} ({time.time()-t0:.0f}s)")
        if not ok:
            all_results[model_id] = {"status": "download_failed"}
            continue

        # Step 2: Extract weights
        print(f"  Extracting weights...")
        try:
            weights = extract_transformer_weights(model_id, device=DEVICE)
        except Exception as e:
            print(f"  SKIP: Extract failed: {e}")
            all_results[model_id] = {"status": "extract_failed", "error": str(e)}
            clear_model_cache(model_id)
            continue

        # Step 3: Create or expand brain
        new_vocab = weights['vocab_size']
        if brain is None:
            print(f"  Creating SOMI brain (vocab={new_vocab})...")
            brain = SOMICircuitBrain(config, input_dim=SOMI_HIDDEN, output_dim=new_vocab)
            brain = brain.to(DEVICE)
            print(f"  Params: {sum(p.numel() for p in brain.parameters())}")
        else:
            current_vocab = brain.y_decoder.out_features
            if new_vocab > current_vocab:
                print(f"  Vocab expansion: {current_vocab} -> {new_vocab}")
                old_state = brain.state_dict()
                brain = SOMICircuitBrain(config, input_dim=SOMI_HIDDEN, output_dim=new_vocab)
                new_state = brain.state_dict()
                for k, v in old_state.items():
                    if k in new_state:
                        if v.shape == new_state[k].shape:
                            new_state[k] = v
                        elif v.dim() >= 1 and v.shape[0] <= new_state[k].shape[0]:
                            slices = [slice(0, s) for s in v.shape]
                            new_state[k][tuple(slices)] = v
                brain.load_state_dict(new_state)
                brain = brain.to(DEVICE)

        # Step 3b: Move extracted weights to DEVICE to avoid CPU/CUDA mismatch
        for layer in weights['layers']:
            for k, v in layer.items():
                if isinstance(v, torch.Tensor):
                    layer[k] = v.to(DEVICE)
        if weights.get('embed_weight') is not None:
            weights['embed_weight'] = weights['embed_weight'].to(DEVICE)
        if weights.get('lm_head_weight') is not None:
            weights['lm_head_weight'] = weights['lm_head_weight'].to(DEVICE)

        # Step 4: Absorb
        fp_before = compute_fingerprint(brain)
        print(f"  Absorbing...")
        try:
            diag = absorb_weights_into_brain(brain, weights, strength=STRENGTH, method=METHOD)
        except Exception as e:
            print(f"  SKIP: Absorb failed: {e}")
            all_results[model_id] = {"status": "absorb_failed", "error": str(e)}
            del weights
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            clear_model_cache(model_id)
            continue

        del weights
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Step 5: Check integrity
        fp_after = compute_fingerprint(brain)
        integrity = check_integrity(brain, verbose=True)
        result = _log_absorption(brain, model_id, size, i, len(MODELS), diag, fp_before, fp_after, integrity)
        all_results[model_id] = result
        absorbed_count += 1

        # Step 6: Checkpoint
        ckpt_path = CHECKPOINT_DIR / f"after_{i}_{model_id.replace('/', '_')}.pt"
        save_checkpoint(brain, str(ckpt_path), step=i)
        # Step 7: Free disk
        clear_model_cache(model_id)

        print(f"  Done. Params={sum(p.numel() for p in brain.parameters())}, "
              f"Absorbed={absorbed_count}/{i}")

    return brain, all_results


def _log_absorption(brain, model_id, size, step, total, diag, fp_before, fp_after, integrity):
    """Log absorption results to console and W&B."""
    healthy = integrity.get('overall_healthy', False)
    energy = diag.get('vocab_svd_energy_preserved', 0)
    vocab_absorbed = diag.get('vocab_absorbed', False)

    spectral_gaps = [v for k, v in fp_after.items() if 'spectral_gap' in k and isinstance(v, float)]
    avg_gap = sum(spectral_gaps) / len(spectral_gaps) if spectral_gaps else 0

    print(f"  Integrity: {'HEALTHY' if healthy else 'ISSUES'}")
    print(f"  Vocab absorbed: {vocab_absorbed}, SVD energy: {energy:.1%}")
    print(f"  Avg spectral gap: {avg_gap:.4f}")

    result = {
        "status": "ok",
        "model": model_id,
        "size": size,
        "healthy": healthy,
        "vocab_absorbed": vocab_absorbed,
        "svd_energy": energy,
        "avg_spectral_gap": avg_gap,
    }

    if HAS_WANDB:
        log_data = {
            "step": step,
            "model": model_id,
            "healthy": int(healthy),
            "svd_energy_preserved": energy,
            "avg_spectral_gap": avg_gap,
            "vocab_absorbed": int(vocab_absorbed),
        }
        for k, v in diag.items():
            if isinstance(v, (int, float)):
                log_data[f"absorb/{k}"] = v
        for k, v in fp_after.items():
            if isinstance(v, float):
                log_data[f"fingerprint/{k}"] = v
        wandb.log(log_data, step=step)

    return result


# ============================================================
# Phase 3: Final Diagnostics
# ============================================================

def final_diagnostics(brain, all_results):
    """Run final diagnostics and save everything."""
    print("\n" + "=" * 70)
    print("  PHASE 3: FINAL DIAGNOSTICS")
    print("=" * 70)

    # Forward pass test
    print("\n  Forward pass test...")
    x = torch.randn(2, 16, SOMI_HIDDEN, device=DEVICE)
    with torch.no_grad():
        logits, brain_diag = brain(x, training=False)
    methods = [v for k, v in brain_diag.items() if 'method' in k]
    print(f"  Output shape: {logits.shape}")
    print(f"  Settle methods: {methods}")
    print(f"  Diagnostics keys: {len(brain_diag)}")

    # Final fingerprint
    fp_final = compute_fingerprint(brain)
    print(f"\n  Final fingerprint:")
    for k, v in sorted(fp_final.items()):
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}")

    # Final integrity
    integrity = check_integrity(brain, verbose=True)
    print(f"  Overall: {'HEALTHY' if integrity['overall_healthy'] else 'ISSUES'}")

    # Save final checkpoint
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    final_path = FINAL_DIR / "somi_absorbed_all.pt"
    save_checkpoint(brain, str(final_path), step=999)
    print(f"\n  Final checkpoint: {final_path}")
    print(f"  File size: {final_path.stat().st_size / 1e6:.1f} MB")

    # Save results JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE,
        "somi_config": {
            "hidden_dim": brain.config.hidden_dim,
            "n_parts": brain.config.n_parts,
        },
        "n_models_attempted": len(MODELS),
        "n_models_absorbed": sum(1 for r in all_results.values() if r.get("status") == "ok"),
        "all_healthy": all(r.get("healthy", False) for r in all_results.values() if r.get("status") == "ok"),
        "final_output_shape": list(logits.shape),
        "results": all_results,
    }
    LOG_FILE.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  Results: {LOG_FILE}")

    if HAS_WANDB:
        wandb.log({
            "final/output_shape": str(list(logits.shape)),
            "final/n_models_absorbed": summary["n_models_absorbed"],
            "final/all_healthy": int(summary["all_healthy"]),
            "final/total_params": sum(p.numel() for p in brain.parameters()),
        })
        wandb.finish()

    return summary


# ============================================================
# Main
# ============================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("  SOMI: ABSORB ALL MODELS")
    print(f"  {len(MODELS)} models -> 1 brain")
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 70)

    brain, all_results = absorb_all()

    if brain is None:
        print("\n  FAILED: No models were absorbed.")
        sys.exit(1)

    summary = final_diagnostics(brain, all_results)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  COMPLETE")
    print(f"  Models absorbed: {summary['n_models_absorbed']}/{len(MODELS)}")
    print(f"  All healthy: {summary['all_healthy']}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 70)


if __name__ == '__main__':
    main()
