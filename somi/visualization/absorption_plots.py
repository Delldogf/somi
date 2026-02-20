"""
SOMI 4.0 Absorption Plots — 5 absorption visualizations
==========================================================

1. Fingerprint radar (before/after knowledge probe scores)
2. Delta W heatmap (what changed during transplant)
3. Domain scores (per-domain performance after absorption)
4. Integrity panel (all integrity checks)
5. Multi-model breakdown (contribution from each specialist)
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional


def plot_fingerprint_comparison(
    before: Dict[str, float],
    after: Dict[str, float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Radar chart comparing fingerprints before and after absorption."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Find matching probe scores
    probes = []
    before_scores = []
    after_scores = []
    for k in sorted(before.keys()):
        if 'confidence' in k and k in after:
            probes.append(k.replace('probe_', 'P').replace('_confidence', ''))
            before_scores.append(before[k])
            after_scores.append(after[k])

    if not probes:
        ax.text(0, 0, 'No probe data', ha='center')
        return fig

    # Radar plot
    n = len(probes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    before_scores += before_scores[:1]
    after_scores += after_scores[:1]

    ax.plot(angles, before_scores, 'b-o', label='Before', linewidth=2)
    ax.fill(angles, before_scores, alpha=0.1, color='blue')
    ax.plot(angles, after_scores, 'r-o', label='After', linewidth=2)
    ax.fill(angles, after_scores, alpha=0.1, color='red')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(probes)
    ax.set_title('Knowledge Fingerprint')
    ax.legend(loc='upper right')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_integrity_panel(
    integrity_report: Dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Show all integrity check results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    parts = []
    checks = []
    for pid, report in integrity_report.items():
        if isinstance(report, dict) and 'all_ok' in report:
            parts.append(pid)
            checks.append(report)

    if not parts:
        ax.text(0.5, 0.5, 'No integrity data', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    check_names = ['W_diag_ok', 'W_nonneg_ok', 'W_row_bounded',
                    'mass_conductivity_ok', 'spectral_gap_ok', 'mass_range_ok']
    data = np.zeros((len(parts), len(check_names)))

    for i, c in enumerate(checks):
        for j, name in enumerate(check_names):
            data[i, j] = 1.0 if c.get(name, False) else 0.0

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(check_names)))
    ax.set_xticklabels([n.replace('_ok', '') for n in check_names], rotation=45, ha='right')
    ax.set_yticks(range(len(parts)))
    ax.set_yticklabels(parts)
    ax.set_title('Integrity Checks (Green=Pass, Red=Fail)')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
