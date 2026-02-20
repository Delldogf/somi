"""
SOMI 4.0 W&B Panel Definitions — 11 panel groups
====================================================

Defines the W&B dashboard layout for monitoring SOMI training.
Call setup_wandb_dashboard() at the start of training to configure panels.

Panel groups:
1.  Overview — loss, accuracy, health score
2.  L1 Physics — stress, mass, settling, W stats
3.  L2 Continuum — mass-conductivity, spectral, scaling
4.  L3 Spacetime — CFL, c_info, dark energy/matter
5.  L4 Gauge — Wilson loops, curvature, Chern-Simons
6.  L5 Quantum — ensemble, topological protection
7.  Neuroscience — E/I, criticality, modularity, Hebbian
8.  Circuit — throughput, coherence, bottleneck, shared Parts
9.  Pathology — pathology count, health score, fixes
10. Training — LM loss, JEPA loss, gradients
11. Absorption — fingerprint scores, integrity, deltas
"""

from typing import Dict


PANEL_DEFINITIONS = {
    'Overview': [
        'total_loss', 'lm_loss', 'health_score', 'n_pathologies',
        'brain_aggregated_magnitude',
    ],
    'L1_Physics': [
        'L1_part*_W_mean', 'L1_part*_mass_mean', 'L1_part*_spectral_gap',
        'L1_part*_arousal', 'L1_part*_W_sparsity', 'L1_part*_W_asymmetry',
    ],
    'L2_Continuum': [
        'L2_part*_mc_violation', 'L2_part*_spectral_entropy',
        'L2_part*_spectral_effective_dim', 'L2_part*_scaling_ratio',
    ],
    'L3_Spacetime': [
        'L3_part*_cfl_satisfied', 'L3_part*_c_info_mean',
        'L3_part*_hawking_temperature', 'L3_part*_dark_matter_ratio',
    ],
    'L4_Gauge': [
        'L4_sys*_wilson_curvature', 'L4_yang_mills_action',
        'L4_chern_simons', 'L4_gauge_smoothness',
    ],
    'L5_Quantum': [
        'L5_ensemble_uncertainty_mean', 'L5_saddle_point_quality',
        'L5_topological_protection', 'L5_rg_scale_ratio',
    ],
    'Neuroscience': [
        'neuro_part*_ei_balance', 'neuro_part*_criticality_index',
        'neuro_part*_modularity_proxy', 'neuro_part*_hebbian_consistency',
        'neuro_part*_small_world_index', 'neuro_part*_clustering_coefficient',
    ],
    'Circuit': [
        'circuit_total_generalization_pressure', 'circuit_bottleneck_part',
        'circuit_bottleneck_arousal', 'circuit_shared*_pressure',
    ],
    'Pathology': [
        'health_score', 'n_pathologies', 'n_fixes_applied',
    ],
    'Training': [
        'lm_loss', 'jepa_prediction_loss', 'jepa_variance_loss',
        'jepa_covariance_loss', 'macro_grad_norm', 'training_step',
    ],
    'Absorption': [
        '*_delta_magnitude', '*_transplant_W_change',
        'probe_*_confidence', 'probe_*_entropy',
    ],
}


def setup_wandb_dashboard():
    """
    Configure W&B dashboard with all 11 panel groups.

    Call this at the start of training. It creates W&B sections
    that organize the ~100+ metrics into logical groups.
    """
    try:
        import wandb
        if wandb.run is None:
            return

        # W&B automatically groups metrics by prefix
        # Just define the summary metrics
        wandb.define_metric('step')
        wandb.define_metric('*', step_metric='step')

        # Log panel definitions for reference
        wandb.config.update({
            'panel_definitions': {k: v for k, v in PANEL_DEFINITIONS.items()},
            'total_panel_groups': len(PANEL_DEFINITIONS),
        })

    except ImportError:
        pass


def get_panel_metrics(diagnostics: Dict, panel_name: str) -> Dict:
    """
    Extract metrics for a specific panel from the full diagnostics dict.

    Args:
        diagnostics: Full diagnostics dict
        panel_name: One of the panel group names

    Returns:
        panel_metrics: Filtered dict with just this panel's metrics
    """
    if panel_name not in PANEL_DEFINITIONS:
        return {}

    patterns = PANEL_DEFINITIONS[panel_name]
    result = {}

    for key, value in diagnostics.items():
        if not isinstance(value, (int, float)):
            continue
        for pattern in patterns:
            if '*' in pattern:
                # Simple wildcard matching
                prefix, suffix = pattern.split('*', 1)
                if key.startswith(prefix) and key.endswith(suffix):
                    result[key] = value
                    break
            elif key == pattern:
                result[key] = value
                break

    return result
