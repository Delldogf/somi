"""
SOMI Universal Absorber — Multi-Teacher Schedule
===================================================

Manages absorbing knowledge from multiple teachers in a stable,
repeatable order with interference prevention.

The schedule:
  1. Fingerprint the current brain
  2. For each teacher (in priority order):
     a. Fingerprint the teacher
     b. knowledge_diff to find what's missing
     c. stress_guided_transplant only where needed
     d. Integrity check — roll back if broken
     e. Re-fingerprint to measure gain
  3. Log everything for the model's lifetime record

Interference prevention:
  - After each teacher, check that previously-absorbed knowledge
    wasn't destroyed (re-run probes from earlier teachers)
  - If interference detected, reduce strength and retry
"""

import torch
from typing import Dict, List, Optional

from .transplant import stress_guided_transplant, spectral_mode_transfer
from .fingerprint import compute_fingerprint, knowledge_diff, compare_fingerprints
from .integrity import check_integrity


class UniversalAbsorber:
    """
    Manages a schedule of teacher absorptions with interference checking.

    Args:
        brain: Target SOMICircuitBrain
        interference_threshold: max acceptable drop in prior teacher's metrics
        max_retries: retries with reduced strength before skipping a teacher
    """

    def __init__(
        self,
        brain: 'SOMICircuitBrain',
        interference_threshold: float = 0.1,
        max_retries: int = 3,
    ):
        self.brain = brain
        self.interference_threshold = interference_threshold
        self.max_retries = max_retries
        self.absorption_log: List[Dict] = []

    def absorb_schedule(
        self,
        teachers: List[Dict],
        probe_inputs: Optional[List[torch.Tensor]] = None,
        probe_labels: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """
        Run the full absorption schedule.

        Args:
            teachers: List of teacher dicts, each containing:
                - 'brain': SOMICircuitBrain (the teacher)
                - 'name': str (for logging)
                - 'strength': float (initial absorption strength, default 1.0)
                - 'method': 'stress_guided' or 'spectral' (default 'stress_guided')
            probe_inputs: Probes for fingerprinting
            probe_labels: Labels for fingerprinting probes

        Returns:
            report: Full absorption report
        """
        report = {'teachers': [], 'total_absorbed': 0, 'total_skipped': 0}

        fp_before_all = compute_fingerprint(
            self.brain, probe_inputs, probe_labels
        )
        report['fingerprint_before'] = fp_before_all

        prior_fingerprints = []

        for i, teacher in enumerate(teachers):
            name = teacher.get('name', f'teacher_{i}')
            strength = teacher.get('strength', 1.0)
            method = teacher.get('method', 'stress_guided')
            teacher_brain = teacher['brain']

            print(f"\n[Absorber] Teacher {i+1}/{len(teachers)}: {name} "
                  f"(method={method}, strength={strength:.2f})")

            fp_before = compute_fingerprint(
                self.brain, probe_inputs, probe_labels
            )
            diff = knowledge_diff(fp_before, compute_fingerprint(
                teacher_brain, probe_inputs, probe_labels
            ))

            teacher_report = {
                'name': name,
                'method': method,
                'knowledge_diff': diff,
            }

            absorbed = False
            for retry in range(self.max_retries):
                current_strength = strength * (0.5 ** retry)

                # Save state for potential rollback
                saved_W = {}
                for pid, part in self.brain.parts.items():
                    saved_W[pid] = part.W_local.clone()

                # Absorb
                if method == 'spectral':
                    self._absorb_spectral(teacher_brain, current_strength)
                else:
                    stress_guided_transplant(
                        self.brain, teacher_brain,
                        strength=current_strength,
                    )

                # Integrity check
                integrity = check_integrity(self.brain, verbose=False)
                if not integrity.get('overall_healthy', False):
                    print(f"  Integrity failed. Rolling back.")
                    for pid, part in self.brain.parts.items():
                        if pid in saved_W:
                            part.W_local.copy_(saved_W[pid])
                    continue

                # Interference check against prior teachers
                fp_after = compute_fingerprint(
                    self.brain, probe_inputs, probe_labels
                )
                interference = self._check_interference(
                    fp_after, prior_fingerprints
                )
                if interference > self.interference_threshold:
                    print(f"  Interference {interference:.3f} > threshold. "
                          f"Retry {retry+1} with strength={current_strength/2:.3f}")
                    for pid, part in self.brain.parts.items():
                        if pid in saved_W:
                            part.W_local.copy_(saved_W[pid])
                    continue

                # Success
                absorbed = True
                gain = compare_fingerprints(fp_before, fp_after)
                teacher_report['absorbed'] = True
                teacher_report['strength_used'] = current_strength
                teacher_report['retries'] = retry
                teacher_report['gain'] = gain
                teacher_report['interference'] = interference
                prior_fingerprints.append(fp_after)
                print(f"  Absorbed successfully (strength={current_strength:.3f}, "
                      f"interference={interference:.3f})")
                break

            if not absorbed:
                teacher_report['absorbed'] = False
                teacher_report['reason'] = 'max_retries_exceeded'
                report['total_skipped'] += 1
                print(f"  Skipped after {self.max_retries} retries.")
            else:
                report['total_absorbed'] += 1

            report['teachers'].append(teacher_report)
            self.absorption_log.append(teacher_report)

        # Recalibrate after all absorptions
        self.brain.recalibrate_config()

        fp_after_all = compute_fingerprint(
            self.brain, probe_inputs, probe_labels
        )
        report['fingerprint_after'] = fp_after_all
        report['overall_gain'] = compare_fingerprints(
            fp_before_all, fp_after_all
        )

        return report

    def _absorb_spectral(
        self, teacher_brain: 'SOMICircuitBrain', strength: float
    ):
        """Absorb via spectral mode transfer."""
        from ..physics.forces import compute_laplacian
        from ..physics.settling import compute_eigendecomposition

        for pid in self.brain.parts:
            if pid not in teacher_brain.parts:
                continue
            teacher_part = teacher_brain.parts[pid]
            L = compute_laplacian(teacher_part.W_local)
            evals, evecs, _ = compute_eigendecomposition(L)
            spectral_mode_transfer(
                self.brain.parts[pid], evals, evecs, strength=strength
            )

    def _check_interference(
        self,
        current_fp: Dict[str, float],
        prior_fps: List[Dict[str, float]],
    ) -> float:
        """Check if absorbing this teacher degraded prior knowledge."""
        if not prior_fps:
            return 0.0

        max_drop = 0.0
        for prior_fp in prior_fps:
            for key in prior_fp:
                if 'confidence' in key and key in current_fp:
                    drop = prior_fp[key] - current_fp[key]
                    max_drop = max(max_drop, drop)
                if 'stress_mean' in key and key in current_fp:
                    increase = current_fp[key] - prior_fp[key]
                    max_drop = max(max_drop, increase)

        return max_drop
