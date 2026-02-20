"""
Test all SOMI features: absorption, training, compression, AIMO, checkpoint.
"""
import torch
import tempfile
import os

from somi.config import SOMIBrainConfig
from somi.brain.circuit_brain import SOMICircuitBrain


def test_brain():
    print("=" * 60)
    print("  1. Basic brain (SSM settling, spectral_K)")
    print("=" * 60)
    config = SOMIBrainConfig.auto(64, 2)
    brain = SOMICircuitBrain(config, input_dim=32, output_dim=50)
    x = torch.randn(2, 4, 32)
    logits, diag = brain(x, training=True)
    methods = [v for k, v in diag.items() if 'method' in k]
    print(f"  spectral_K={config.spectral_K}, settle methods={methods}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Diagnostics: {len(diag)} metrics")
    return brain, config


def test_fingerprint(brain):
    print("\n" + "=" * 60)
    print("  2. Full fingerprint (stress + eigen + Wilson)")
    print("=" * 60)
    from somi.absorption import compute_fingerprint, knowledge_diff
    fp = compute_fingerprint(brain)
    stress_keys = [k for k in fp if 'stress' in k]
    eigen_keys = [k for k in fp if 'spectral' in k or 'eigen' in k]
    wilson_keys = [k for k in fp if 'wilson' in k]
    print(f"  Stress signals: {len(stress_keys)}")
    print(f"  Spectral signals: {len(eigen_keys)}")
    print(f"  Wilson signals: {len(wilson_keys)}")
    print(f"  Total fingerprint keys: {len(fp)}")
    return fp


def test_knowledge_diff(brain):
    print("\n" + "=" * 60)
    print("  3. Knowledge diff (what does B know that A doesn't?)")
    print("=" * 60)
    from somi.absorption import compute_fingerprint, knowledge_diff
    brain2 = SOMICircuitBrain(brain.config, input_dim=32, output_dim=50)
    fp_a = compute_fingerprint(brain)
    fp_b = compute_fingerprint(brain2)
    diff = knowledge_diff(fp_a, fp_b)
    print(f"  Parts needing knowledge: {diff.get('parts_needing_knowledge', [])}")
    print(f"  Diff keys: {len(diff)}")


def test_stress_guided_transplant(brain):
    print("\n" + "=" * 60)
    print("  4. Stress-guided transplant")
    print("=" * 60)
    from somi.absorption import stress_guided_transplant
    source = SOMICircuitBrain(brain.config, input_dim=32, output_dim=50)
    diag = stress_guided_transplant(brain, source, strength=0.5)
    guided = diag.get('stress_guided', False)
    coverage = [v for k, v in diag.items() if 'mask_coverage' in k]
    print(f"  Stress guided: {guided}")
    print(f"  Mask coverage per part: {coverage}")


def test_spectral_transfer(brain):
    print("\n" + "=" * 60)
    print("  5. Spectral mode transfer")
    print("=" * 60)
    from somi.absorption import spectral_mode_transfer
    from somi.physics.forces import compute_laplacian
    from somi.physics.settling import compute_eigendecomposition
    source = SOMICircuitBrain(brain.config, input_dim=32, output_dim=50)
    src_part = source.parts['0']
    L = compute_laplacian(src_part.W_local)
    evals, evecs, _ = compute_eigendecomposition(L)
    diag = spectral_mode_transfer(brain.parts['0'], evals, evecs, strength=0.3)
    print(f"  K modes transferred: {diag['spectral_transfer_K']}")
    print(f"  W change: {diag['spectral_transfer_W_change']:.6f}")


def test_integrity(brain):
    print("\n" + "=" * 60)
    print("  6. Integrity check (with Wilson loops)")
    print("=" * 60)
    from somi.absorption import check_integrity
    report = check_integrity(brain, verbose=False)
    print(f"  Overall healthy: {report['overall_healthy']}")
    print(f"  Wilson healthy: {report.get('wilson_healthy', 'N/A')}")


def test_dual_learning(brain, config):
    print("\n" + "=" * 60)
    print("  7. Full JEPA pipeline (DualLearningTrainer)")
    print("=" * 60)
    from somi.training import DualLearningTrainer
    trainer = DualLearningTrainer(
        brain, config, lr=1e-3,
        use_jepa=True, jepa_weight=0.1,
        selective_threshold=0.1,
        mass_guided=True,
    )
    x = torch.randn(2, 4, 32)
    ids = torch.randint(0, 50, (2, 4))
    loss, diag = trainer.step(ids, x, ids, training=True, phase=2)
    print(f"  Total loss: {loss:.4f}")
    print(f"  LM loss: {diag.get('lm_loss', 'N/A'):.4f}")
    print(f"  JEPA loss: {diag.get('jepa_total_loss', 'N/A'):.4f}")
    print(f"  Stress-JEPA corr: {diag.get('stress_jepa_correlation', 'N/A')}")
    print(f"  Phase: {diag.get('training_phase', 'N/A')}")
    print(f"  Selective coverage: {diag.get('selective_coverage', 'N/A')}")


def test_stress_sampler():
    print("\n" + "=" * 60)
    print("  8. Stress-based data sampler")
    print("=" * 60)
    from somi.training import StressDataSampler
    sampler = StressDataSampler(dataset_size=100, batch_size=8)
    indices = sampler.sample()
    sampler.update_stress(indices, {'stress': 0.8})
    indices2 = sampler.sample()
    sampler.update_stress(indices2, {'stress': 0.1})
    stats = sampler.get_stats()
    print(f"  Coverage: {stats['coverage']:.0%}")
    print(f"  Mean stress: {stats['mean_stress']:.3f}")


def test_auto_compress(brain):
    print("\n" + "=" * 60)
    print("  9. AutoCompress")
    print("=" * 60)
    from somi.compression import AutoCompress
    comp = AutoCompress(brain, patience=3, cooldown=1)
    comp.stress_ema = 0.05
    comp.spectral_ratio_ema = 0.3
    for _ in range(5):
        comp.step({'stress': 0.02, 'eigen_used_modes': 3, 'eigen_total_modes': 64})
    print(f"  Events: {len(comp.compress_events)}")
    print(f"  Status: stress={comp.stress_ema:.3f}, spectral={comp.spectral_ratio_ema:.3f}")


def test_checkpoint(brain):
    print("\n" + "=" * 60)
    print("  10. Checkpoint save/load")
    print("=" * 60)
    from somi.checkpoint import save_checkpoint, load_checkpoint, record_event
    history = []
    record_event(history, 'create', step=0, details={'hidden_dim': 64})
    record_event(history, 'grow', step=100, details={'from': 64, 'to': 96})

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        model_id = save_checkpoint(brain, path, step=100, history=history)
        brain2, meta = load_checkpoint(path, input_dim=32, output_dim=50)
        print(f"  Model ID: {model_id}")
        print(f"  Loaded step: {meta.get('step', 'N/A')}")
        print(f"  History events: {len(meta.get('history', []))}")
        print(f"  Loaded H={brain2.config.hidden_dim}, P={brain2.config.n_parts}")
        x = torch.randn(2, 4, 32)
        out, _ = brain2(x, training=False)
        print(f"  Forward pass after load: {out.shape}")
    finally:
        os.unlink(path)


def test_aimo(brain):
    print("\n" + "=" * 60)
    print("  11. AIMO 3 pipeline")
    print("=" * 60)
    from somi.aimo import consistency_score, settle_check, confidence_consistency_fusion

    states = [torch.randn(2, 64) for _ in range(5)]
    score, diag = consistency_score(states)
    print(f"  Consistency score: {score:.3f}")

    x = torch.randn(2, 4, 32)
    stability, sdiag = settle_check(brain, x)
    print(f"  Settle stability: {stability:.3f}")

    candidates = [
        {'confidence': 0.9, 'consistency': 0.8, 'stability': 0.7},
        {'confidence': 0.6, 'consistency': 0.95, 'stability': 0.9},
        {'confidence': 0.7, 'consistency': 0.5, 'stability': 0.4},
    ]
    best, fdiag = confidence_consistency_fusion(candidates, alpha=0.4)
    print(f"  Best candidate: {best} (score={fdiag['fusion_best_score']:.3f})")
    print(f"  Scores: {[f'{s:.3f}' for s in fdiag['fusion_scores']]}")


if __name__ == '__main__':
    brain, config = test_brain()
    test_fingerprint(brain)
    test_knowledge_diff(brain)
    test_stress_guided_transplant(brain)
    test_spectral_transfer(brain)
    test_integrity(brain)
    test_dual_learning(brain, config)
    test_stress_sampler()
    test_auto_compress(brain)
    test_checkpoint(brain)
    test_aimo(brain)

    print("\n" + "=" * 60)
    print("  ALL 11 TESTS PASSED")
    print("=" * 60)
