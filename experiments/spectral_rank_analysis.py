"""
Spectral Rank Analysis — Phase 1 of Lossless Absorption
==========================================================

Downloads transformer models and computes Marchenko-Pastur spectral analysis
to determine:
1. How many signal modes exist per layer (effective knowledge dimensions)
2. What fraction of weights are noise vs signal
3. The minimum SOMI hidden_dim needed for lossless absorption

This answers the fundamental question: how big does SOMI need to be?
"""

import sys
import os
import json
import time
import torch
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from somi.absorption.from_huggingface import extract_transformer_weights
from somi.absorption.spectral_analysis import (
    analyze_full_model,
    print_analysis_report,
    marchenko_pastur_threshold,
)

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clear_hf_cache(model_id):
    """Clear cached model files to free disk."""
    cache_dir = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub",
    )
    model_dir_name = "models--" + model_id.replace("/", "--")
    model_path = os.path.join(cache_dir, model_dir_name)
    if os.path.exists(model_path):
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fnames in os.walk(model_path)
            for f in fnames
        ) / 1e6
        shutil.rmtree(model_path, ignore_errors=True)
        return size_mb
    return 0


def main():
    print(f"\n{'='*70}")
    print(f"  MARCHENKO-PASTUR SPECTRAL RANK ANALYSIS")
    print(f"  Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    print(f"  Models to analyze: {len(MODELS)}")
    print(f"{'='*70}\n")

    all_summaries = []
    global_max_signal = 0
    global_max_eff_rank = 0
    t_start = time.time()

    for idx, (model_id, label) in enumerate(MODELS, 1):
        print(f"\n--- Model {idx}/{len(MODELS)}: {model_id} ({label}) ---")
        t0 = time.time()

        try:
            weights = extract_transformer_weights(model_id, device=DEVICE)

            summary = analyze_full_model(weights)
            print_analysis_report(summary)

            all_summaries.append({
                'model': model_id,
                'label': label,
                'hidden_dim': summary['hidden_dim'],
                'n_layers': summary['n_layers'],
                'vocab_size': summary['vocab_size'],
                'signal_modes': summary.get('signal_modes_per_layer'),
                'effective_rank': summary.get('effective_rank_per_layer'),
                'entropy_rank': summary.get('entropy_rank_per_layer'),
                'recommended_hidden_dim': summary.get('recommended_hidden_dim'),
                'embed_signal_modes': summary['embed_analysis']['K_signal'] if summary.get('embed_analysis') else None,
                'embed_effective_rank': summary['embed_analysis']['effective_rank'] if summary.get('embed_analysis') else None,
            })

            if summary.get('signal_modes_per_layer'):
                global_max_signal = max(global_max_signal,
                                        summary['signal_modes_per_layer']['max'])
            if summary.get('effective_rank_per_layer'):
                global_max_eff_rank = max(global_max_eff_rank,
                                          summary['effective_rank_per_layer']['max'])

            del weights
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"  ERROR: {e}")
            all_summaries.append({'model': model_id, 'label': label, 'error': str(e)})

        freed = clear_hf_cache(model_id)
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.0f}s, freed {freed:.0f} MB cache")

    total_time = time.time() - t_start

    # Final summary
    print(f"\n{'='*70}")
    print(f"  GLOBAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Total models analyzed: {len(all_summaries)}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"\n  Max signal modes across all layers/models: {global_max_signal}")
    print(f"  Max effective rank across all layers/models: {global_max_eff_rank:.0f}")

    # Compute recommended SOMI sizes
    def next_64(n):
        return ((n + 63) // 64) * 64

    rec_from_signal = next_64(int(global_max_signal * 1.1))
    rec_from_effrank = next_64(int(global_max_eff_rank * 1.1))

    print(f"\n  RECOMMENDED SOMI HIDDEN_DIM:")
    print(f"    From MP signal modes:  {rec_from_signal}")
    print(f"    From effective rank:   {rec_from_effrank}")
    print(f"    Conservative choice:   {max(rec_from_signal, rec_from_effrank)}")

    print(f"\n  Per-model breakdown:")
    for s in all_summaries:
        if 'error' in s:
            print(f"    {s['label']:6s} ({s['model']:40s}): ERROR - {s['error'][:60]}")
        else:
            sm = s.get('signal_modes', {})
            er = s.get('effective_rank', {})
            rec = s.get('recommended_hidden_dim', '?')
            print(f"    {s['label']:6s} (H={s['hidden_dim']:5d}, L={s['n_layers']:2d}): "
                  f"signal_max={sm.get('max','?'):>4}, "
                  f"eff_rank_max={er.get('max',0):>6.0f}, "
                  f"recommended={rec}")

    print(f"{'='*70}\n")

    # Save results
    results_path = "/workspace/spectral_analysis_results.json" if os.path.exists("/workspace") else "spectral_analysis_results.json"
    with open(results_path, "w") as f:
        serializable = []
        for s in all_summaries:
            s_clean = {k: v for k, v in s.items()}
            serializable.append(s_clean)
        json.dump(serializable, f, indent=2, default=str)
    print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
