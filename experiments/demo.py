"""
SOMI 4.0 End-to-End Demo
===========================

This script demonstrates the ENTIRE SOMI 4.0 pipeline:

1. Create a Circuit-S brain (4 Parts, 2 Systems)
2. Forward pass with random input (no training needed)
3. Run ALL 70+ diagnostics via DiagnosticDashboard
4. Show self-healing in action (if any pathologies detected)
5. Generate brain scan visualization
6. Print health summary

Run with: python -m demo
"""

import torch

from somi.config import SOMIBrainConfig
from somi.brain.circuit_brain import SOMICircuitBrain
from somi.diagnostics.dashboard import DiagnosticDashboard
from somi.absorption.integrity import check_integrity


def run_demo():
    """Run the full SOMI 4.0 demonstration."""
    print("=" * 70)
    print("  SOMI 4.0 — Self-Organizing Models of Intelligence")
    print("  End-to-End Demo")
    print("=" * 70)
    print()

    # 1. Create brain
    print("[1/6] Creating Circuit-S brain...")
    config = SOMIBrainConfig.circuit_s()
    brain = SOMICircuitBrain(config, input_dim=64, output_dim=100)

    n_params = sum(p.numel() for p in brain.parameters())
    n_buffers = sum(b.numel() for b in brain.buffers())
    print(f"  Architecture: {config.n_parts} Parts, {config.n_systems} Systems")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  System routes: {config.system_routes}")
    print(f"  Shared Parts: {config.shared_part_ids}")
    print(f"  Learnable params: {n_params:,}")
    print(f"  Physics buffers: {n_buffers:,}")
    print()

    # 2. Forward pass
    print("[2/6] Running forward pass...")
    x = torch.randn(2, 8, 64)  # batch=2, seq=8, dim=64
    logits, fwd_diag = brain(x, training=True)
    print(f"  Input shape:  {list(x.shape)}")
    print(f"  Output shape: {list(logits.shape)}")
    print(f"  Diagnostics collected: {len(fwd_diag)} metrics")
    print()

    # 3. Run full diagnostics
    print("[3/6] Running ALL diagnostics...")
    dashboard = DiagnosticDashboard(config, auto_heal=True, wandb_log=False)
    report = dashboard.report(brain, step=0, test_input=x)

    health = report['health_score']
    n_path = report['n_pathologies']
    print(f"  Health Score: {health:.0f}/100")
    print(f"  Pathologies detected: {n_path}")
    if report.get('pathologies'):
        for p in report['pathologies']:
            print(f"    - [{p['severity']}] {p['name']}: {p['description']}")
    print()

    # 4. Self-healing
    fixes = report.get('fixes_applied', [])
    if fixes:
        print(f"[4/6] Self-healing applied {len(fixes)} fix(es):")
        for f in fixes:
            print(f"    Part {f['part_id']}: {f['action']}")
    else:
        print("[4/6] No self-healing needed (brain is healthy)")
    print()

    # 5. Integrity check
    print("[5/6] Running integrity checks...")
    integrity = check_integrity(brain, verbose=True)
    print(f"  Overall: {'HEALTHY' if integrity['overall_healthy'] else 'ISSUES DETECTED'}")
    print()

    # 6. Per-Part summary
    print("[6/6] Part-by-Part Summary:")
    print("-" * 50)
    for pid, part in brain.parts.items():
        shared = "SHARED" if int(pid) in (config.shared_part_ids or []) else "      "
        print(f"  Part {pid} [{shared}]:")
        print(f"    Mass: mean={part.mass.mean():.4f}, std={part.mass.std():.4f}")
        print(f"    W: mean={part.W_local.abs().mean():.4f}, sparsity={(part.W_local.abs() < 1e-6).float().mean():.1%}")
        print(f"    Arousal: {part.arousal.item():.3f}")
        print(f"    Settle steps: {part.n_settle}")
        print(f"    Eta: {part.eta:.6f}")
    print()

    # 7. Generate visualization (if matplotlib available)
    try:
        from somi.visualization.brain_scan import plot_brain_scan
        from somi.visualization.circuit_plots import plot_circuit_flow

        print("Generating visualizations...")
        fig1 = plot_brain_scan(brain, report, save_path='brain_scan.png')
        fig2 = plot_circuit_flow(brain, save_path='circuit_flow.png')
        print("  Saved: brain_scan.png, circuit_flow.png")
    except Exception as e:
        print(f"  Visualization skipped: {e}")

    print()
    print("=" * 70)
    print("  SOMI 4.0 Demo Complete!")
    print(f"  Health: {health:.0f}/100 | Parts: {config.n_parts} | "
          f"Systems: {config.n_systems} | Diagnostics: {len(report)} metrics")
    print("=" * 70)

    return brain, report


if __name__ == '__main__':
    brain, report = run_demo()
