"""
SOMI Auto-Growth: Stress-Triggered Neurogenesis
=================================================

Monitors the stress across all layers during training. When the model
is struggling (high stress for sustained periods), it automatically
triggers neurogenesis — adding new neurons to expand capacity.

This implements the biological principle: when existing neural circuits
are overloaded, the brain grows new neurons to handle the demand.

Usage:
    growth_monitor = AutoGrowth(model, max_hidden=1024, growth_factor=1.5)
    for step, batch in enumerate(dataloader):
        logits, loss, diagnostics = model(batch, training=True)
        growth_monitor.step(diagnostics)  # may trigger growth
"""

import torch
from typing import Optional


class AutoGrowth:
    """
    Monitors model stress and triggers neurogenesis when needed.

    The trigger logic:
    1. Collect average stress from all layers each step
    2. Track a running average of stress
    3. If stress stays above threshold for patience steps, grow
    4. After growth, reset the counter (give the model time to use new neurons)

    Args:
        model: SOMILanguageModel instance
        stress_threshold: stress level above which growth is considered
        patience: how many consecutive high-stress steps before growing
        growth_factor: multiply hidden_dim by this when growing (e.g., 1.5 = 50% more neurons)
        max_hidden: maximum hidden dimension (stop growing past this)
        cooldown: steps to wait after a growth event before considering again
    """

    def __init__(
        self,
        model: 'SOMILanguageModel',
        stress_threshold: float = 0.5,
        patience: int = 100,
        growth_factor: float = 1.5,
        max_hidden: int = 1024,
        cooldown: int = 200,
    ):
        self.model = model
        self.stress_threshold = stress_threshold
        self.patience = patience
        self.growth_factor = growth_factor
        self.max_hidden = max_hidden
        self.cooldown = cooldown

        self.high_stress_count = 0
        self.steps_since_growth = 0
        self.growth_events = []
        self.stress_history = []
        self.stress_ema = 0.0

    def step(self, diagnostics: dict) -> bool:
        """
        Check stress and potentially trigger growth.

        Args:
            diagnostics: the diagnostics dict from model.forward()

        Returns:
            True if growth was triggered this step
        """
        self.steps_since_growth += 1

        stress_values = [
            v for k, v in diagnostics.items()
            if 'stress' in k and isinstance(v, (int, float))
        ]
        if not stress_values:
            return False

        avg_stress = sum(stress_values) / len(stress_values)
        self.stress_history.append(avg_stress)

        alpha = 0.95
        self.stress_ema = alpha * self.stress_ema + (1 - alpha) * avg_stress

        if self.stress_ema > self.stress_threshold:
            self.high_stress_count += 1
        else:
            self.high_stress_count = max(0, self.high_stress_count - 1)

        if (
            self.high_stress_count >= self.patience
            and self.steps_since_growth >= self.cooldown
            and self.model.hidden_dim < self.max_hidden
        ):
            return self._trigger_growth()

        return False

    def _trigger_growth(self) -> bool:
        """Execute neurogenesis."""
        old_H = self.model.hidden_dim
        new_H = min(int(old_H * self.growth_factor), self.max_hidden)

        if new_H <= old_H:
            return False

        print(f"\n[AutoGrowth] Stress threshold exceeded for {self.high_stress_count} steps")
        print(f"[AutoGrowth] Growing model: H={old_H} -> H={new_H}")

        self.model.grow(new_H)

        self.growth_events.append({
            'step': len(self.stress_history),
            'old_hidden': old_H,
            'new_hidden': new_H,
            'stress_at_trigger': self.stress_ema,
        })

        self.high_stress_count = 0
        self.steps_since_growth = 0

        print(f"[AutoGrowth] Growth complete. New hidden_dim={self.model.hidden_dim}")
        return True

    def get_status(self) -> dict:
        return {
            'current_hidden': self.model.hidden_dim,
            'max_hidden': self.max_hidden,
            'stress_ema': self.stress_ema,
            'high_stress_count': self.high_stress_count,
            'growth_events': len(self.growth_events),
            'steps_since_growth': self.steps_since_growth,
        }


class AutoGrowthFull:
    """
    Full auto-growth: grows neurons, adds Parts, adds Systems, recalibrates.

    This extends AutoGrowth with structural growth: when shared Parts are
    saturated (high stress on shared Parts, low on specialized), the brain
    adds a new Part and wires it into a new System.  After any growth event,
    the action-derived parameters are recalibrated for the new size.

    Theory: hidden_dim and n_parts should BOTH be automatic.  The model
    starts from config.from_scratch() (tiny) and evolves its own size.

    Monitors:
      1. Overall stress -> grow hidden_dim (same as AutoGrowth)
      2. Shared-Part stress vs specialized-Part stress -> add Part
      3. System saturation -> add System through new + shared Parts
      4. After any structural change -> recalibrate physics params

    Args:
        brain: SOMICircuitBrain instance
        stress_threshold: stress above which neuron growth is considered
        part_stress_ratio: shared/specialized stress ratio to trigger add_part
        patience: consecutive high-stress steps before growing
        growth_factor: multiply hidden_dim by this when growing neurons
        max_hidden: stop growing neurons past this
        max_parts: stop adding Parts past this
        cooldown: steps after a growth event before considering again
    """

    def __init__(
        self,
        brain: 'SOMICircuitBrain',
        stress_threshold: float = 0.5,
        part_stress_ratio: float = 2.0,
        patience: int = 100,
        growth_factor: float = 1.5,
        max_hidden: int = 1024,
        max_parts: int = 32,
        cooldown: int = 200,
    ):
        self.brain = brain
        self.stress_threshold = stress_threshold
        self.part_stress_ratio = part_stress_ratio
        self.patience = patience
        self.growth_factor = growth_factor
        self.max_hidden = max_hidden
        self.max_parts = max_parts
        self.cooldown = cooldown

        self.high_stress_count = 0
        self.high_shared_stress_count = 0
        self.steps_since_growth = 0
        self.growth_events = []
        self.stress_ema = 0.0
        self.shared_stress_ema = 0.0
        self.specialized_stress_ema = 0.0

    def step(self, diagnostics: dict) -> str:
        """
        Check stress and potentially trigger growth.

        Returns:
            'none', 'neurons', 'part', or 'system' indicating what grew.
        """
        self.steps_since_growth += 1

        stress_values = [
            v for k, v in diagnostics.items()
            if 'stress' in k and isinstance(v, (int, float))
        ]
        if not stress_values:
            return 'none'

        avg_stress = sum(stress_values) / len(stress_values)
        alpha = 0.95
        self.stress_ema = alpha * self.stress_ema + (1 - alpha) * avg_stress

        shared_ids = set(str(s) for s in (self.brain.config.shared_part_ids or []))
        shared_stresses = [
            v for k, v in diagnostics.items()
            if 'stress' in k and isinstance(v, (int, float))
            and any(f'part_{sid}' in k for sid in shared_ids)
        ]
        specialized_stresses = [
            v for k, v in diagnostics.items()
            if 'stress' in k and isinstance(v, (int, float))
            and not any(f'part_{sid}' in k for sid in shared_ids)
        ]

        if shared_stresses:
            avg_shared = sum(shared_stresses) / len(shared_stresses)
            self.shared_stress_ema = alpha * self.shared_stress_ema + (1 - alpha) * avg_shared
        if specialized_stresses:
            avg_spec = sum(specialized_stresses) / len(specialized_stresses)
            self.specialized_stress_ema = alpha * self.specialized_stress_ema + (1 - alpha) * avg_spec

        if self.steps_since_growth < self.cooldown:
            return 'none'

        if self.stress_ema > self.stress_threshold:
            self.high_stress_count += 1
        else:
            self.high_stress_count = max(0, self.high_stress_count - 1)

        ratio = (self.shared_stress_ema / max(self.specialized_stress_ema, 1e-8))
        if ratio > self.part_stress_ratio:
            self.high_shared_stress_count += 1
        else:
            self.high_shared_stress_count = max(0, self.high_shared_stress_count - 1)

        if (
            self.high_shared_stress_count >= self.patience
            and self.brain.config.n_parts < self.max_parts
        ):
            return self._add_part_and_system()

        if (
            self.high_stress_count >= self.patience
            and self.brain.config.hidden_dim < self.max_hidden
        ):
            return self._grow_neurons()

        return 'none'

    def _grow_neurons(self) -> str:
        old_H = self.brain.config.hidden_dim
        new_H = min(int(old_H * self.growth_factor), self.max_hidden)
        if new_H <= old_H:
            return 'none'

        print(f"\n[AutoGrowthFull] Growing neurons: H={old_H} -> H={new_H}")
        self.brain.grow_brain(new_H)
        self.brain.recalibrate_config()

        self.growth_events.append({
            'type': 'neurons',
            'step': self.steps_since_growth,
            'old_hidden': old_H,
            'new_hidden': new_H,
            'stress': self.stress_ema,
        })
        self._reset_counters()
        print(f"[AutoGrowthFull] Done. Recalibrated for H={new_H}, P={self.brain.config.n_parts}")
        return 'neurons'

    def _add_part_and_system(self) -> str:
        old_P = self.brain.config.n_parts
        print(f"\n[AutoGrowthFull] Shared Parts saturated (ratio={self.shared_stress_ema/max(self.specialized_stress_ema,1e-8):.2f})")
        print(f"[AutoGrowthFull] Adding Part: P={old_P} -> P={old_P + 1}")

        new_part_id = self.brain.add_part()

        shared_id = (self.brain.config.shared_part_ids or [0])[0]
        route = [shared_id, new_part_id]
        new_sys_id = self.brain.add_system(route)

        self.brain.recalibrate_config()

        self.growth_events.append({
            'type': 'part+system',
            'step': self.steps_since_growth,
            'new_part_id': new_part_id,
            'new_system_id': new_sys_id,
            'route': route,
            'shared_stress': self.shared_stress_ema,
            'specialized_stress': self.specialized_stress_ema,
        })
        self._reset_counters()
        print(f"[AutoGrowthFull] Done. Part {new_part_id}, System {new_sys_id} (route {route})")
        print(f"[AutoGrowthFull] Recalibrated for H={self.brain.config.hidden_dim}, P={self.brain.config.n_parts}")
        return 'part'

    def _reset_counters(self):
        self.high_stress_count = 0
        self.high_shared_stress_count = 0
        self.steps_since_growth = 0

    def get_status(self) -> dict:
        return {
            'hidden_dim': self.brain.config.hidden_dim,
            'n_parts': self.brain.config.n_parts,
            'n_systems': len(self.brain.systems),
            'stress_ema': self.stress_ema,
            'shared_stress_ema': self.shared_stress_ema,
            'specialized_stress_ema': self.specialized_stress_ema,
            'growth_events': len(self.growth_events),
            'steps_since_growth': self.steps_since_growth,
        }
