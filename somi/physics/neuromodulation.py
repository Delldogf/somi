"""
SOMI 2.0 Neuromodulation
==========================

The brain has four major neuromodulatory systems that globally regulate
neural computation. Each one acts as a "volume knob" that adjusts a
different aspect of processing:

    NE (norepinephrine)  → Alertness    → How fast to learn
    DA (dopamine)        → Reward       → Should I learn more or less
    ACh (acetylcholine)  → Attention    → Which features to process faster
    5-HT (serotonin)     → Patience     → How long to think

In the brain, these aren't separate design choices — they're consequences
of the neural architecture. Each neuromodulator is released by a small
nucleus (cluster of neurons) that monitors a specific global signal and
broadcasts its modulation across the whole cortex.

In SOMI, we do the same:
    1. Each "nucleus" monitors a computable signal from the model's state
    2. It outputs a level (0 to 1 or a per-feature vector)
    3. That level modulates an existing SOMI parameter
    4. NO NEW FREE PARAMETERS — everything is derived from internal state

Brain correspondence:
    NE  ← Locus coeruleus      → monitors surprise
    DA  ← Ventral tegmental area → monitors reward prediction error
    ACh ← Basal forebrain       → monitors salience / uncertainty
    5-HT ← Dorsal raphe         → monitors difficulty / frustration

Source:
    Doc 25 (Neuroscience Engineering Manual)
    Predictions P31-P35 (ACh from SOMI_2_0_COMPLETE_THEORY.md)
    Schlingloff et al. 2026, Chuang et al. 2025, Li et al. 2025,
    Bouabid et al. 2026 (ACh papers)
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NeuromodulatorState:
    """
    Current levels of all four neuromodulators.

    Each level is between 0 and 1 (or a per-feature vector for ACh).
    These are computed from the model's internal state each forward pass.

    Think of this as a blood sample showing current neurotransmitter levels.
    """
    ne_level: float = 0.5       # Norepinephrine (arousal/alertness)
    da_level: float = 0.5       # Dopamine (reward signal)
    ach_salience: Optional[torch.Tensor] = None  # Per-feature ACh [hidden_dim]
    serotonin_level: float = 0.5  # Serotonin (patience)


class NeuromodulatorSystem(nn.Module):
    """
    The four neuromodulatory nuclei of SOMI.

    Each nucleus:
        1. Has a small amount of internal state (running averages)
        2. Computes its level from the model's current state
        3. Returns modulated parameters (eta, M_vector, n_settle, noise)

    This class does NOT learn via backprop — it's all running averages
    and deterministic functions, just like the real brainstem nuclei.

    No new free parameters:
        - NE: uses existing arousal computation
        - DA: derived from stress improvement (reward = geometry helping)
        - ACh: derived from per-feature error magnitude (salience)
        - 5-HT: derived from relative difficulty (error vs running average)

    Brain analog:
        The brainstem and basal forebrain contain small nuclei (~100-1000 neurons
        each) that broadcast neuromodulators across the entire cortex. They don't
        process information — they regulate HOW information is processed. SOMI's
        NeuromodulatorSystem is exactly this: a small module that monitors global
        signals and adjusts processing parameters.
    """

    def __init__(self, hidden_dim: int, ei_ratio: float = 0.8):
        super().__init__()

        # === Running averages for computing neuromodulator levels ===
        # These are exponential moving averages (EMAs) that smooth the
        # raw signals, just like real neurotransmitter reuptake smooths
        # the neural signals.

        # NE: already handled by arousal system in calibration.py
        # (we just read the arousal value from the layer)

        # DA: track stress improvement (reward prediction error)
        self.register_buffer('stress_ema', torch.tensor(1.0))
        self.register_buffer('da_level', torch.tensor(0.5))

        # ACh: track per-feature salience (error magnitude)
        self.register_buffer('salience_ema', torch.ones(hidden_dim))
        self.register_buffer('ach_salience', torch.ones(hidden_dim))

        # 5-HT: track difficulty (error relative to expectation)
        self.register_buffer('difficulty_ema', torch.tensor(1.0))
        self.register_buffer('serotonin_level', torch.tensor(0.5))

        # Store ei_ratio for ACh modulation strength
        self._ei_ratio = ei_ratio

    def update(
        self,
        error: torch.Tensor,       # [batch, seq, hidden] prediction error
        stress: float,              # Current stress value
        arousal: float,             # From existing arousal system (= NE)
    ) -> NeuromodulatorState:
        """
        Update all neuromodulator levels from current model state.

        This is called once per forward pass, after settling and before
        geometry update. Each nucleus computes its level from a different
        global signal.

        Args:
            error:   Prediction error from the settled field
            stress:  Current information stress (from stress tensor)
            arousal: Current arousal level (from existing NE/arousal system)

        Returns:
            NeuromodulatorState with current levels
        """
        with torch.no_grad():
            # =========================================================
            # NE (Norepinephrine) — Alertness
            # =========================================================
            # Already computed by the arousal system in calibration.py.
            # We just read it and pass it through.
            #
            # Brain: Locus coeruleus monitors ACC/OFC for surprise.
            #        Phasic NE = focused attention on unexpected events.
            #        Tonic NE = broad alertness when utility is low.
            ne_level = arousal  # Range: [0.01, 0.99]

            # =========================================================
            # DA (Dopamine) — Reward Prediction Error
            # =========================================================
            # Signal: Is the geometry HELPING? (stress going down = reward)
            #
            # Brain: VTA computes RPE = actual_reward - expected_reward.
            #        Positive RPE = "that was better than expected" → learn more.
            #        Negative RPE = "that was worse than expected" → learn less.
            #
            # In SOMI: reward = stress improvement (geometry is doing its job)
            # RPE = (expected_stress - actual_stress) / expected_stress
            #     = (stress_ema - stress) / stress_ema
            # Positive = stress decreased (good!) → DA high → learn more
            # Negative = stress increased (bad) → DA low → learn less
            #
            # No new parameters: DA is derived from stress trajectory.

            stress_ema_val = self.stress_ema.item()
            if stress_ema_val > 1e-8:
                rpe = (stress_ema_val - stress) / stress_ema_val
            else:
                rpe = 0.0

            # Sigmoid squash: DA level ∈ (0, 1)
            # Clamp to prevent math overflow with large stress values
            rpe_clamped = max(-10.0, min(10.0, -3.0 * rpe))
            new_da = 1.0 / (1.0 + math.exp(rpe_clamped))
            # The 3.0 scaling makes DA respond to ~30% stress changes.
            # This is NOT a hyperparameter — it's the slope of the sigmoid,
            # analogous to the gain of DA neurons (~3-5x for significant RPEs).

            # Smooth update (slow reuptake, tau ~ 20 steps)
            self.da_level.fill_(0.95 * self.da_level.item() + 0.05 * new_da)

            # Update stress EMA (slow baseline, tau ~ 50 steps)
            self.stress_ema.fill_(0.98 * stress_ema_val + 0.02 * stress)

            da_level = self.da_level.item()

            # =========================================================
            # ACh (Acetylcholine) — Attention / Salience
            # =========================================================
            # Signal: Which features have unusually high prediction error?
            #
            # Brain: Basal forebrain releases ACh based on uncertainty/salience.
            #        Phasic ACh (P31): transiently reduces M_i → faster processing
            #        of salient features.
            #        Think: "Hey, pay attention to THIS feature — something's
            #        happening there."
            #
            # In SOMI: salience_i = |error_i| / mean(|error_j|)
            #   - salience > 1 = this feature has more error than average
            #   - salience < 1 = this feature is fine
            #   ACh reduces M_i for salient features → they oscillate faster
            #   → process information faster → correct errors faster
            #
            # Modulation strength = (1 - ei_ratio) = fraction of I neurons
            # This is principled: ACh is primarily released near I neurons
            # (basal forebrain cholinergic neurons project to I interneurons)
            #
            # Validated by: P31 (Schlingloff et al. 2026) — phasic ACh
            # transiently reduces M_i at salient events.

            # Per-feature error magnitude (averaged over batch and sequence)
            error_mag = error.detach().abs().mean(dim=(0, 1))  # [hidden_dim]

            # Salience = relative error (mean-normalized)
            mean_error = error_mag.mean().clamp(min=1e-8)
            salience = error_mag / mean_error  # Mean = 1.0

            # Smooth update (moderate reuptake, tau ~ 10 steps)
            self.salience_ema.mul_(0.9).add_(0.1 * salience)

            # ACh salience: how much each feature stands out
            # Clamp to [0.5, 2.0] to prevent extreme modulation
            self.ach_salience.copy_(self.salience_ema.clamp(min=0.5, max=2.0))

            ach_salience = self.ach_salience.clone()

            # =========================================================
            # 5-HT (Serotonin) — Patience / Persistence
            # =========================================================
            # Signal: How hard is the current problem?
            #
            # Brain: Dorsal raphe 5-HT neurons fire more when the animal
            #        needs to WAIT for reward (patience). Low 5-HT = impulsive
            #        (give up quickly). High 5-HT = patient (keep thinking).
            #
            # In SOMI: difficulty = error / running_average
            #   - difficulty > 1 = harder than usual → think longer (more n_settle)
            #   - difficulty < 1 = easier than usual → think less (fewer n_settle)
            #
            # No new parameters: 5-HT is derived from error trajectory.
            #
            # Additional effect: 5-HT modulates noise (exploration).
            #   High 5-HT (hard problem) → slightly more noise → explore more
            #   (SSRIs increase serotonin → increase exploration)

            # Current difficulty relative to baseline
            error_mean = error.detach().abs().mean().item()
            difficulty_ema_val = self.difficulty_ema.item()

            if difficulty_ema_val > 1e-8:
                relative_difficulty = error_mean / difficulty_ema_val
            else:
                relative_difficulty = 1.0

            # 5-HT level: sigmoid of relative difficulty
            # > 0.5 when problem is hard, < 0.5 when easy
            sht_arg = max(-10.0, min(10.0, -2.0 * (relative_difficulty - 1.0)))
            new_5ht = 1.0 / (1.0 + math.exp(sht_arg))
            # The 2.0 scaling makes 5-HT respond to ~50% difficulty changes.

            # Smooth update (slow reuptake, tau ~ 30 steps)
            self.serotonin_level.fill_(
                0.97 * self.serotonin_level.item() + 0.03 * new_5ht
            )

            # Update difficulty EMA (very slow baseline, tau ~ 100 steps)
            self.difficulty_ema.fill_(
                0.99 * difficulty_ema_val + 0.01 * error_mean
            )

            serotonin_level = self.serotonin_level.item()

        return NeuromodulatorState(
            ne_level=ne_level,
            da_level=da_level,
            ach_salience=ach_salience,
            serotonin_level=serotonin_level,
        )

    def modulate_eta(
        self,
        eta_base: float,
        state: NeuromodulatorState,
    ) -> float:
        """
        Modulate geometry learning rate with NE and DA.

        NE (arousal) effect: already applied in calibration.py via
            eta = eta_base * (0.5 + arousal)
        We DON'T double-apply NE here.

        DA effect: reward prediction error modulates learning.
            - High DA (good reward) → learn 1.5x faster (reinforce what's working)
            - Low DA (bad reward) → learn 0.5x faster (don't reinforce failure)

        Combined: eta_final = eta_ne * da_multiplier
            where da_multiplier = 0.5 + da_level (range: [0.5, 1.5])

        Brain analog:
            DA from VTA modulates LTP/LTD in cortex.
            High DA → more LTP (strengthen useful connections).
            Low DA → less LTP (stop reinforcing bad patterns).

        Args:
            eta_base: Current eta (already modulated by NE/arousal)
            state:    Current neuromodulator state

        Returns:
            eta_modulated: Final geometry learning rate
        """
        # DA modulation: [0.5, 1.5]x
        da_multiplier = 0.5 + state.da_level
        return eta_base * da_multiplier

    def modulate_mass(
        self,
        M_vector: torch.Tensor,
        state: NeuromodulatorState,
    ) -> torch.Tensor:
        """
        Modulate per-feature mass with ACh (attention).

        ACh REDUCES mass of salient features → they oscillate faster →
        process information faster → correct errors faster.

        Formula:
            M_i_modulated = M_i * (1 - strength * (salience_i - 1).clamp(min=0))

        where:
            strength = (1 - ei_ratio) = fraction of I neurons (principled: ACh
            is released near I interneurons in the basal forebrain)

            salience_i = |error_i| / mean(|error_j|) (smoothed)

        Effect:
            - salience = 1.0 (average): no change
            - salience = 2.0 (twice average error): mass reduced by (1-ei_ratio)
              At ei_ratio = 0.8: mass reduced by 20%
            - salience = 0.5 (half average error): no change (ACh doesn't INCREASE mass)

        Brain analog (P31): Phasic ACh transiently reduces effective membrane
        capacitance at salient locations, making those neurons respond faster.

        Args:
            M_vector: [hidden_dim] per-feature mass from geometric inertia
            state:    Current neuromodulator state

        Returns:
            M_modulated: [hidden_dim] ACh-modulated mass
        """
        if state.ach_salience is None:
            return M_vector

        # Strength of ACh modulation = fraction of I neurons
        # This is principled: ACh's cortical effects are mediated through
        # inhibitory interneurons (basket cells, etc.)
        strength = 1.0 - self._ei_ratio  # 0.2 at default 80/20

        # How much to reduce mass: proportional to EXCESS salience
        # (salience - 1).clamp(min=0) is 0 for normal features, positive for salient
        excess_salience = (state.ach_salience - 1.0).clamp(min=0.0)

        # Modulation factor: 1.0 for normal features, down to (1 - strength) for very salient
        # Clamp to prevent mass from going too low
        modulation = (1.0 - strength * excess_salience).clamp(min=0.5)

        return M_vector * modulation

    def modulate_n_settle(
        self,
        n_settle_base: int,
        state: NeuromodulatorState,
    ) -> int:
        """
        Modulate settling duration with 5-HT (patience).

        High 5-HT (hard problem) → think longer → more settling steps.
        Low 5-HT (easy problem) → think less → fewer settling steps.

        Formula:
            n_settle_modulated = n_settle_base * (0.75 + 0.5 * serotonin_level)

        Range: [0.75x, 1.25x] of base n_settle
            - 5-HT = 0.0 (very easy): 75% of base (save compute on easy problems)
            - 5-HT = 0.5 (normal): 100% of base (no change)
            - 5-HT = 1.0 (very hard): 125% of base (think longer)

        Brain analog:
            5-HT from dorsal raphe promotes waiting behavior. Animals with
            high 5-HT are more patient (wait for larger delayed reward).
            Low 5-HT → impulsive (choose immediate small reward).

            In SOMI: "patience" = number of settling steps. More steps =
            more time for oscillatory dynamics to find a good solution.

        Args:
            n_settle_base: Current n_settle (from eigenfrequency calibration)
            state:         Current neuromodulator state

        Returns:
            n_settle_modulated: Adjusted number of settling steps
        """
        # 5-HT modulation: [0.75x, 1.25x]
        multiplier = 0.75 + 0.5 * state.serotonin_level
        return max(3, int(n_settle_base * multiplier))

    def modulate_noise(
        self,
        noise_ratio_base: float,
        state: NeuromodulatorState,
    ) -> float:
        """
        Modulate exploration noise with 5-HT.

        High 5-HT (hard problem) → slightly more noise → explore more solutions.
        This is the SSRI effect: increased serotonin → increased exploration.

        Formula:
            noise_modulated = noise_base * (0.8 + 0.4 * serotonin_level)

        Range: [0.8x, 1.2x]
            - 5-HT = 0.0: 80% noise (less exploration on easy problems)
            - 5-HT = 0.5: 100% noise (no change)
            - 5-HT = 1.0: 120% noise (more exploration on hard problems)

        Brain analog:
            SSRIs (which increase 5-HT) reduce repetitive behaviors and
            increase cognitive flexibility — essentially more exploration.
            5-HT modulates the gain of noise in cortical circuits.

        Args:
            noise_ratio_base: Base noise ratio (regime choice)
            state:            Current neuromodulator state

        Returns:
            noise_modulated: Adjusted noise ratio
        """
        multiplier = 0.8 + 0.4 * state.serotonin_level
        return noise_ratio_base * multiplier

    def get_diagnostics(self, state: NeuromodulatorState) -> Dict[str, float]:
        """
        Get neuromodulator diagnostics for logging.

        Returns a dict with all neuromodulator levels and derived metrics.
        """
        diag = {
            'neuro_ne': state.ne_level,
            'neuro_da': state.da_level,
            'neuro_5ht': state.serotonin_level,
        }

        if state.ach_salience is not None:
            diag['neuro_ach_mean'] = state.ach_salience.mean().item()
            diag['neuro_ach_max'] = state.ach_salience.max().item()
            diag['neuro_ach_min'] = state.ach_salience.min().item()

        # Derived metrics
        diag['neuro_da_eta_multiplier'] = 0.5 + state.da_level
        diag['neuro_5ht_settle_multiplier'] = 0.75 + 0.5 * state.serotonin_level
        diag['neuro_5ht_noise_multiplier'] = 0.8 + 0.4 * state.serotonin_level

        return diag
