"""
SOMI 4.0 Pathology Detection
===============================

11 named failure modes with detection, severity, and recommended fixes.
This is SOMI's "doctor" — it can diagnose specific problems and prescribe
specific treatments. Transformers can only say "loss went up."

Pathology Table:
  1.  hamiltonian_increasing — dH/dt > 0 (physics broken)
  2.  geometry_explosion — |W| growing unbounded
  3.  spectral_collapse — spectral gap -> 0 (all modes same)
  4.  mass_uniformity — all masses equal (no hierarchy)
  5.  stress_divergence — stress -> infinity
  6.  oscillation_death — phi_dot -> 0 too fast
  7.  arousal_saturation — arousal stuck at 0 or 1
  8.  ei_imbalance — E/I ratio far from 80/20
  9.  settling_timeout — phi doesn't converge
  10. connectivity_death — W -> 0 (all connections pruned)
  11. frequency_collapse — all eigenmodes same frequency
"""

import torch
from typing import Dict, List, Tuple


# Pathology table: name -> (detection_condition, severity, healthy_range, fix)
PATHOLOGY_TABLE = {
    'hamiltonian_increasing': {
        'description': 'Energy is INCREASING during settling (should always decrease)',
        'severity': 'CRITICAL',
        'check': lambda d: d.get('hamiltonian_violation_rate', 0) > 0.1,
        'healthy_range': 'violation_rate < 0.01',
        'fix': 'Reduce dt, increase damping (target_zeta), check CFL condition',
    },
    'geometry_explosion': {
        'description': 'W magnitude growing out of control',
        'severity': 'CRITICAL',
        'check': lambda d: d.get('W_magnitude', 0) > 5.0,
        'healthy_range': 'W_mean < 2.0',
        'fix': 'Increase lambda_W (weight decay), increase timescale_ratio',
    },
    'spectral_collapse': {
        'description': 'Spectral gap approaching zero (all eigenmodes identical)',
        'severity': 'HIGH',
        'check': lambda d: d.get('spectral_gap', 1) < 0.01,
        'healthy_range': 'spectral_gap > 0.05',
        'fix': 'Reduce eta, increase structural plasticity, check sparsity',
    },
    'mass_uniformity': {
        'description': 'All features have the same mass (no hierarchy)',
        'severity': 'MEDIUM',
        'check': lambda d: d.get('mass_std', 1) < 0.01,
        'healthy_range': 'mass_cv > 0.1',
        'fix': 'Ensure Herfindahl mass computation is running, check W diversity',
    },
    'stress_divergence': {
        'description': 'Stress tensor growing without bound',
        'severity': 'HIGH',
        'check': lambda d: d.get('stress_magnitude', 0) > 100.0,
        'healthy_range': 'stress_mean < 10.0',
        'fix': 'Increase stress_momentum_beta, reduce eta, check input normalization',
    },
    'oscillation_death': {
        'description': 'Velocity dying too fast (over-damped)',
        'severity': 'MEDIUM',
        'check': lambda d: d.get('velocity_norm', 1) < 1e-6,
        'healthy_range': 'velocity_norm > 1e-4',
        'fix': 'Reduce target_zeta (less damping), increase noise_ratio',
    },
    'arousal_saturation': {
        'description': 'Arousal stuck at extreme (0 or 1)',
        'severity': 'MEDIUM',
        'check': lambda d: d.get('arousal', 0.5) < 0.01 or d.get('arousal', 0.5) > 0.99,
        'healthy_range': '0.1 < arousal < 0.9',
        'fix': 'Check error_running_avg, reset arousal to 0.5',
    },
    'ei_imbalance': {
        'description': 'Excitation/inhibition ratio far from target',
        'severity': 'MEDIUM',
        'check': lambda d: abs(d.get('ei_balance', 0.8) - 0.8) > 0.2,
        'healthy_range': '0.6 < E/(E+I) < 0.9',
        'fix': 'Re-run signed_sinkhorn, check ei_mask',
    },
    'settling_timeout': {
        'description': 'Phi not converging within n_settle steps',
        'severity': 'LOW',
        'check': lambda d: d.get('velocity_norm', 0) > 0.5,
        'healthy_range': 'final_velocity < 0.1',
        'fix': 'Increase n_settle, check dt vs eigenfrequencies',
    },
    'connectivity_death': {
        'description': 'W approaching zero (all connections pruned)',
        'severity': 'HIGH',
        'check': lambda d: d.get('W_magnitude', 1) < 0.001,
        'healthy_range': 'W_mean > 0.01',
        'fix': 'Reduce lambda_W, increase noise in structural plasticity',
    },
    'frequency_collapse': {
        'description': 'All eigenmodes have same frequency (no diversity)',
        'severity': 'MEDIUM',
        'check': lambda d: d.get('freq_range', 1) < 0.01,
        'healthy_range': 'freq_range > 0.1',
        'fix': 'Check W initialization, ensure sparsity creates frequency diversity',
    },
}


def detect_pathologies(
    diagnostics: Dict[str, float],
) -> List[Dict[str, str]]:
    """
    Scan diagnostics for pathologies.

    Args:
        diagnostics: Flat dict of all diagnostic values

    Returns:
        List of detected pathologies with name, severity, and fix
    """
    detected = []

    for name, info in PATHOLOGY_TABLE.items():
        try:
            if info['check'](diagnostics):
                detected.append({
                    'name': name,
                    'severity': info['severity'],
                    'description': info['description'],
                    'fix': info['fix'],
                    'healthy_range': info['healthy_range'],
                })
        except Exception:
            pass  # Skip checks that fail (missing keys)

    return detected


def get_health_score(pathologies: List[Dict]) -> float:
    """
    Compute overall health score (0-100) from detected pathologies.

    100 = perfect health, no issues
    0 = critical failure

    Scoring:
    - CRITICAL: -30 points each
    - HIGH: -15 points each
    - MEDIUM: -7 points each
    - LOW: -3 points each
    """
    score = 100.0
    severity_costs = {'CRITICAL': 30, 'HIGH': 15, 'MEDIUM': 7, 'LOW': 3}

    for p in pathologies:
        cost = severity_costs.get(p['severity'], 5)
        score -= cost

    return max(0.0, score)
