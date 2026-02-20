"""
Absorb All Models: Multi-Model Absorption into One SOMI Brain
===============================================================

Downloads 10 open-source LLMs in parallel, then absorbs them
sequentially into a single growing SOMI brain.

Usage (on RunPod or any GPU machine):
    export HF_HOME=/workspace/.cache/huggingface
    python -m experiments.absorb_all_models

The script:
  Phase 1 — Downloads all models in parallel (background processes)
  Phase 2 — Absorbs each model into SOMI (smallest first)
  Phase 3 — Runs diagnostics and saves final checkpoint
"""

import os
import sys
import time
import json
import subprocess
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
from somi.checkpoint import save_checkpoint, record_event

# Optional: W&B
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ============================================================
# Configuration
# ============================================================

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "0.5B"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B"),
    ("stabilityai/stablelm-2-1_6b", "1.6B"),
    ("google/gemma-2-2b", "2B"),
    ("microsoft/phi-2", "2.7B"),
    ("Qwen/Qwen2.5-3B", "3B"),
    ("meta-llama/Llama-3.2-3B", "3B"),
    ("mistralai/Mistral-7B-v0.3", "7B"),
    ("Qwen/Qwen2.5-7B", "7B"),
    ("meta-llama/Llama-3.1-8B", "8B"),
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
# Phase 1: Parallel Download
# ============================================================

def download_all_models():
    """Launch parallel downloads for all models. Returns list of processes."""
    print("\n" + "=" * 70)
    print("  PHASE 1: DOWNLOADING ALL MODELS IN PARALLEL")
    print("=" * 70)

    processes = []
    for model_id, size in MODELS:
        print(f"  Starting download: {model_id} ({size})...")
        env = os.environ.copy()
        proc = subprocess.Popen(
            [sys.executable, "-c",
             f"from huggingface_hub import snapshot_download; "
             f"snapshot_download('{model_id}', ignore_patterns=['*.gguf', '*.bin.index.json'])"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        processes.append((model_id, size, proc))

    return processes


def wait_for_downloads(processes):
    """Wait for all downloads to complete. Returns list of (model_id, success)."""
    print("\n  Waiting for downloads to complete...")
    results = []
    for model_id, size, proc in processes:
        proc.wait()
        success = proc.returncode == 0
        status = "OK" if success else f"FAILED (exit {proc.returncode})"
        if not success:
            stderr = proc.stderr.read().decode()[-500:]
            print(f"  {model_id}: {status}")
            print(f"    Error: {stderr}")
        else:
            print(f"  {model_id} ({size}): {status}")
        results.append((model_id, size, success))
    return results


# ============================================================
# Phase 2: Sequential Absorption
# ============================================================

def absorb_all(download_results):
    """Absorb each successfully downloaded model into SOMI."""
    print("\n" + "=" * 70)
    print("  PHASE 2: ABSORBING INTO SOMI")
    print("=" * 70)

    available = [(mid, size) for mid, size, ok in download_results if ok]
    if not available:
        print("  ERROR: No models downloaded successfully!")
        return None, {}

    print(f"  Models available: {len(available)}/{len(MODELS)}")
    print(f"  SOMI config: H={SOMI_HIDDEN}, P={SOMI_PARTS}, device={DEVICE}")
    print(f"  Method: {METHOD}, Strength: {STRENGTH}")

    # Create initial SOMI brain — use first model's vocab size
    print(f"\n  Extracting first model to determine vocab size...")
    first_weights = extract_transformer_weights(
        available[0][0], device=DEVICE
    )
    vocab_size = first_weights['vocab_size']
    print(f"  Vocab size: {vocab_size}")

    config = SOMIBrainConfig.auto(SOMI_HIDDEN, SOMI_PARTS)
    brain = SOMICircuitBrain(config, input_dim=SOMI_HIDDEN, output_dim=vocab_size)
    brain = brain.to(DEVICE)
    print(f"  SOMI brain created: {sum(p.numel() for p in brain.parameters())} parameters")

    # Init W&B
    if HAS_WANDB:
        wandb.init(
            project="somi-absorb-all",
            name=f"absorb-{len(available)}-models-{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "somi_hidden": SOMI_HIDDEN,
                "somi_parts": SOMI_PARTS,
                "strength": STRENGTH,
                "method": METHOD,
                "n_models": len(available),
                "models": [m for m, _ in available],
                "device": DEVICE,
            },
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    brain_events = []

    # Absorb first model (already extracted)
    fp_before = compute_fingerprint(brain)
    print(f"\n{'='*60}")
    print(f"  [{1}/{len(available)}] Absorbing: {available[0][0]} ({available[0][1]})")
    print(f"{'='*60}")

    diag = absorb_weights_into_brain(brain, first_weights, strength=STRENGTH, method=METHOD)
    del first_weights
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    fp_after = compute_fingerprint(brain)
    integrity = check_integrity(brain, verbose=True)

    result = _log_absorption(
        brain, available[0][0], available[0][1], 1, len(available),
        diag, fp_before, fp_after, integrity
    )
    all_results[available[0][0]] = result
    brain_events.append({"event": "absorb", "model": available[0][0], "step": 1})

    ckpt_path = CHECKPOINT_DIR / f"after_{1}_{available[0][0].replace('/', '_')}.pt"
    save_checkpoint(brain, str(ckpt_path), step=1)
    record_event(brain, "absorb", {"source": available[0][0]})

    # Absorb remaining models
    for i, (model_id, size) in enumerate(available[1:], start=2):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(available)}] Absorbing: {model_id} ({size})")
        print(f"{'='*60}")

        fp_before = compute_fingerprint(brain)

        try:
            weights = extract_transformer_weights(model_id, device=DEVICE)
        except Exception as e:
            print(f"  SKIP: Failed to extract {model_id}: {e}")
            all_results[model_id] = {"status": "extract_failed", "error": str(e)}
            continue

        # Handle vocab size changes — use max of current and new
        new_vocab = weights['vocab_size']
        current_vocab = brain.y_decoder.out_features
        if new_vocab > current_vocab:
            print(f"  Vocab expansion: {current_vocab} -> {new_vocab}")
            old_state = brain.state_dict()
            brain = SOMICircuitBrain(
                brain.config, input_dim=SOMI_HIDDEN, output_dim=new_vocab
            )
            brain.load_state_dict(old_state, strict=False)
            brain = brain.to(DEVICE)

        try:
            diag = absorb_weights_into_brain(
                brain, weights, strength=STRENGTH, method=METHOD
            )
        except Exception as e:
            print(f"  SKIP: Failed to absorb {model_id}: {e}")
            all_results[model_id] = {"status": "absorb_failed", "error": str(e)}
            del weights
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            continue

        del weights
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        fp_after = compute_fingerprint(brain)
        integrity = check_integrity(brain, verbose=True)

        result = _log_absorption(
            brain, model_id, size, i, len(available),
            diag, fp_before, fp_after, integrity
        )
        all_results[model_id] = result
        brain_events.append({"event": "absorb", "model": model_id, "step": i})

        ckpt_path = CHECKPOINT_DIR / f"after_{i}_{model_id.replace('/', '_')}.pt"
        save_checkpoint(brain, str(ckpt_path), step=i)
        record_event(brain, "absorb", {"source": model_id})

        print(f"  SOMI state: H={config.hidden_dim}, P={config.n_parts}, "
              f"params={sum(p.numel() for p in brain.parameters())}")

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
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 70)

    # Phase 1: Download
    processes = download_all_models()
    download_results = wait_for_downloads(processes)

    download_time = time.time() - start_time
    print(f"\n  Downloads complete in {download_time:.0f}s")

    # Phase 2: Absorb
    absorb_start = time.time()
    brain, all_results = absorb_all(download_results)

    if brain is None:
        print("\n  FAILED: No models were absorbed.")
        sys.exit(1)

    absorb_time = time.time() - absorb_start
    print(f"\n  Absorption complete in {absorb_time:.0f}s")

    # Phase 3: Diagnostics
    summary = final_diagnostics(brain, all_results)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  COMPLETE")
    print(f"  Models absorbed: {summary['n_models_absorbed']}/{len(MODELS)}")
    print(f"  All healthy: {summary['all_healthy']}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Download: {download_time:.0f}s, Absorb: {absorb_time:.0f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
