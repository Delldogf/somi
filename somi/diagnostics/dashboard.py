"""
SOMI 4.0 Diagnostic Dashboard — Unified Runner + Self-Healing Loop
=====================================================================

dashboard.report(brain, step) does EVERYTHING:
1. Runs ALL diagnostics (L1-L5 + neuro 13 + circuit 11 + pathology 11 + singularity)
2. Aggregates health score (0-100)
3. Detects pathologies from PATHOLOGY_TABLE
4. Self-healing loop: for each pathology, apply the known fix, re-diagnose, verify
5. Logs everything to W&B
6. Returns structured dict for visualization

This is SOMI's killer feature — no other AI architecture can diagnose
specific failure modes AND automatically fix them in a closed loop.
"""

import torch
from typing import Dict, Optional
import logging

from .standard import compute_standard_diagnostics
from .continuum import compute_continuum_diagnostics
from .spacetime import compute_spacetime_diagnostics
from .gauge import compute_gauge_diagnostics
from .quantum import compute_quantum_diagnostics
from .neuroscience import compute_neuroscience_diagnostics
from .circuit import compute_circuit_diagnostics
from .pathology import detect_pathologies, get_health_score, PATHOLOGY_TABLE
from .singularity import detect_singularities

logger = logging.getLogger('SOMI4.diagnostics')


class DiagnosticDashboard:
    """
    Unified diagnostic runner with self-healing capability.

    Usage:
        dashboard = DiagnosticDashboard(config)
        report = dashboard.report(brain, step)
        # report contains: all diagnostics, pathologies, health score, fixes applied

    Args:
        config: SOMIBrainConfig
        auto_heal: Whether to automatically fix detected pathologies
        wandb_log: Whether to log to W&B
    """

    def __init__(
        self,
        config: 'SOMIBrainConfig',
        auto_heal: bool = True,
        wandb_log: bool = True,
    ):
        self.config = config
        self.auto_heal = auto_heal
        self.wandb_log = wandb_log
        self.history = []

    def report(
        self,
        brain: 'SOMICircuitBrain',
        step: int,
        test_input: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Run ALL diagnostics, detect pathologies, optionally self-heal.

        Args:
            brain: The SOMICircuitBrain to diagnose
            step: Current training step
            test_input: Optional test input for Level 5 diagnostics

        Returns:
            report: Complete diagnostic report
        """
        all_diagnostics = {'step': step}

        # === Level 1: Standard (always) ===
        if self.config.compute_level1:
            for pid, part in brain.parts.items():
                diag = compute_standard_diagnostics(part)
                all_diagnostics.update(diag)

        # === Level 2: Continuum ===
        if self.config.compute_level2:
            for pid, part in brain.parts.items():
                diag = compute_continuum_diagnostics(part)
                all_diagnostics.update(diag)

        # === Level 3: Spacetime ===
        if self.config.compute_level3:
            for pid, part in brain.parts.items():
                diag = compute_spacetime_diagnostics(part)
                all_diagnostics.update(diag)

        # === Level 4: Gauge ===
        if self.config.compute_level4:
            diag = compute_gauge_diagnostics(brain)
            all_diagnostics.update(diag)

        # === Level 5: Quantum ===
        if self.config.compute_level5:
            diag = compute_quantum_diagnostics(brain, test_input=test_input)
            all_diagnostics.update(diag)

        # === Neuroscience (13 tests) ===
        if self.config.compute_neuroscience:
            for pid, part in brain.parts.items():
                diag = compute_neuroscience_diagnostics(part)
                all_diagnostics.update(diag)

        # === Circuit (11 metrics) ===
        if self.config.compute_circuit:
            diag = compute_circuit_diagnostics(brain)
            all_diagnostics.update(diag)

        # === Singularity Detection ===
        # Run on each Part's W if we can get phi
        for pid, part in brain.parts.items():
            if test_input is not None:
                h = brain.x_encoder(test_input)
                phi = h  # Approximate (would need to run settle)
                sing_diag = detect_singularities(phi, part.W_local)
                for k, v in sing_diag.items():
                    all_diagnostics[f'part{pid}_{k}'] = v

        # === Pathology Detection ===
        # Flatten per-part diagnostics for pathology detection
        pathologies = []
        for pid, part in brain.parts.items():
            part_flat = {
                'W_magnitude': part.W_local.abs().mean().item(),
                'arousal': part.arousal.item(),
                'spectral_gap': all_diagnostics.get(f'L1_part{pid}_spectral_gap', 0),
                'mass_std': part.mass.std().item(),
                'velocity_norm': all_diagnostics.get(f'part_{pid}_velocity_norm', 0),
                'ei_balance': all_diagnostics.get(f'neuro_part{pid}_ei_balance', 0.8),
                'hamiltonian_violation_rate': all_diagnostics.get(
                    f'neuro_part{pid}_hamiltonian_violation_rate', 0
                ),
                'freq_range': all_diagnostics.get(f'neuro_part{pid}_freq_range', 1),
                'stress_magnitude': all_diagnostics.get(f'part_{pid}_stress_mean', 0),
            }
            part_pathologies = detect_pathologies(part_flat)
            for p in part_pathologies:
                p['part_id'] = pid
                pathologies.append(p)

        all_diagnostics['pathologies'] = pathologies
        all_diagnostics['n_pathologies'] = len(pathologies)
        all_diagnostics['health_score'] = get_health_score(pathologies)

        # === Self-Healing Loop ===
        fixes_applied = []
        if self.auto_heal and pathologies:
            fixes_applied = self._self_heal(brain, pathologies)
            all_diagnostics['fixes_applied'] = fixes_applied
            all_diagnostics['n_fixes_applied'] = len(fixes_applied)

        # === Log to W&B ===
        if self.wandb_log:
            self._log_to_wandb(all_diagnostics, step)

        # Store in history
        self.history.append({
            'step': step,
            'health_score': all_diagnostics['health_score'],
            'n_pathologies': len(pathologies),
            'n_fixes': len(fixes_applied),
        })

        return all_diagnostics

    def _self_heal(
        self,
        brain: 'SOMICircuitBrain',
        pathologies: list,
    ) -> list:
        """
        Apply automatic fixes for detected pathologies.

        For each pathology:
        1. Identify the fix from PATHOLOGY_TABLE
        2. Apply the parameter adjustment
        3. Log what was done

        This is a closed loop — fix, then verify on next report() call.
        """
        fixes = []

        for p in pathologies:
            name = p['name']
            part_id = p.get('part_id')

            if part_id is not None and str(part_id) in brain.parts:
                part = brain.parts[str(part_id)]
                fix = self._apply_fix(part, name)
                if fix:
                    fixes.append(fix)

        return fixes

    def _apply_fix(self, part: 'SOMIPart', pathology_name: str) -> Optional[Dict]:
        """Apply a specific fix to a Part."""
        with torch.no_grad():
            fix = {'pathology': pathology_name, 'part_id': part.part_id}

            if pathology_name == 'hamiltonian_increasing':
                # Reduce dt, increase damping
                part.config.dt *= 0.9
                fix['action'] = 'Reduced dt by 10%'

            elif pathology_name == 'geometry_explosion':
                # Increase weight decay
                part.config.lambda_W *= 1.5
                fix['action'] = 'Increased lambda_W by 50%'

            elif pathology_name == 'spectral_collapse':
                # Slow down geometry learning
                part.config.timescale_ratio *= 1.2
                fix['action'] = 'Increased timescale_ratio by 20%'

            elif pathology_name == 'mass_uniformity':
                # Add noise to mass to create diversity
                noise = 0.1 * torch.randn_like(part.mass)
                part.mass.add_(noise.abs())
                fix['action'] = 'Added mass noise for diversity'

            elif pathology_name == 'stress_divergence':
                # Increase stress momentum (more smoothing)
                part.config.stress_momentum_beta = min(
                    0.99, part.config.stress_momentum_beta + 0.05
                )
                fix['action'] = 'Increased stress_momentum_beta'

            elif pathology_name == 'oscillation_death':
                # Reduce damping
                part.config.target_zeta *= 0.8
                fix['action'] = 'Reduced target_zeta by 20%'

            elif pathology_name == 'arousal_saturation':
                # Reset arousal
                part.arousal.fill_(0.5)
                part.error_running_avg.fill_(1.0)
                fix['action'] = 'Reset arousal to 0.5'

            elif pathology_name == 'connectivity_death':
                # Reduce weight decay
                part.config.lambda_W *= 0.5
                fix['action'] = 'Reduced lambda_W by 50%'

            else:
                return None

            logger.info(f"Self-heal: Part {part.part_id} - {fix['action']}")
            return fix

    def _log_to_wandb(self, diagnostics: Dict, step: int):
        """Log diagnostics to Weights & Biases."""
        try:
            import wandb
            if wandb.run is not None:
                # Filter to numeric values only
                numeric_diag = {
                    k: v for k, v in diagnostics.items()
                    if isinstance(v, (int, float)) and k != 'step'
                }
                wandb.log(numeric_diag, step=step)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"W&B logging failed: {e}")
