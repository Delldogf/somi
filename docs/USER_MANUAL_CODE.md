# SOMI Code — User Manual

This manual explains **how to install, run, and use the SOMI codebase**. For the theory behind SOMI, see [USER_MANUAL_THEORY.md](USER_MANUAL_THEORY.md).

---

## 1. What This Repo Is

- **One clean SOMI package:** `somi` — no path hacks, no dependency on other workspaces.
- **Engine:** config + physics + brain + LM + absorption + compression + diagnostics + visualization + training + utils.
- **Experiments:** Runnable scripts that use the engine (demos, tests, train, absorb, grow).

**Quick orientation:** [START_HERE_WHAT_IS_THIS.md](../START_HERE_WHAT_IS_THIS.md).

---

## 2. Install

From the repo root (the folder that contains `pyproject.toml` and the `somi` package):

```bash
pip install -e .
```

**Requirements:** Python 3.10+, PyTorch. See `pyproject.toml` for full dependencies.

---

## 3. Run Something (First Steps)

All commands from repo root (or with repo on `PYTHONPATH`).

**Demo (recommended first run):**

```bash
python -m experiments.demo
```

This builds a Circuit-S brain, runs a forward pass, runs diagnostics, and shows a brain scan.

**Tests:**

```bash
python -m experiments.test_e2e      # Full E2E: LM, growth, distillation, generation
python -m experiments.test_growth  # Part growth and behavior preservation
```

**Other experiments:**

```bash
python -m experiments.simulate_lifecycle   # Conception → gestation → birth
python -m experiments.absorb_and_grow      # Absorb from another model + grow
python -m experiments.train_addition       # Toy addition task (macro + micro learning)
```

---

## 4. Config (How to Build a Brain)

Everything starts from **`SOMIBrainConfig`** in `somi.config`.

**Presets (fixed sizes):**

```python
from somi.config import SOMIBrainConfig

config = SOMIBrainConfig.circuit_s()   # ~2M params, 4 Parts, 2 Systems, hidden=128
config = SOMIBrainConfig.circuit_m()   # ~15M, 8 Parts, 4 Systems, hidden=256
config = SOMIBrainConfig.circuit_l()   # ~100M, 16 Parts, 8 Systems, hidden=512
config = SOMIBrainConfig.circuit_xl()  # ~400M, 32 Parts, 16 Systems, hidden=1024
```

**Zero-hyperparameter mode (recommended):**

```python
config = SOMIBrainConfig.auto(hidden_dim=128, n_parts=4)
```

This uses the **full Level 1-5 constraint cascade** (`somi/physics/action_derived.py`) to derive **every parameter** from the single SOMI action given only `hidden_dim` and `n_parts`: coupling, damping, dt, weight decay, noise, gate, timescales, EMAs, sparsity, white-matter rank, and feature flags. Runtime calibration (calibration.py) then further sets beta, n_settle, eta, mass from the actual W. **Zero free parameters.** Run `python -m experiments.show_action_cascade` to see what it derives at every scale.

**True zero-input mode (start from nothing and grow):**

```python
config = SOMIBrainConfig.from_scratch()  # H=16, P=1, everything else from physics
```

This starts at the smallest viable brain. Pair with `AutoGrowthFull` to let it grow neurons, add Parts, and add Systems automatically from stress.

**Then build the brain:**

```python
from somi.brain.circuit_brain import SOMICircuitBrain

brain = SOMICircuitBrain(config, input_dim=64, output_dim=100)
# Forward:
logits, diagnostics = brain(x, training=True)  # x: (batch, seq, input_dim)
```

---

## 5. Main Packages and Entry Points

| Package / path | Purpose | Main entry points |
|----------------|---------|-------------------|
| **`somi.config`** | Brain and run config | `SOMIBrainConfig`, presets, `auto()` |
| **`somi.physics`** | Field + geometry dynamics | `forces.py` (9 forces), `settling.py` (SSM/symplectic), `geometry.py`, `calibration.py` |
| **`somi.brain`** | Circuit brain | `SOMICircuitBrain`, `Part`, `System`, `WhiteMatter` |
| **`somi.lm`** | Language model use | `SOMILanguageModel`, `create_somi_lm`, `Distiller`, `AutoGrowth` |
| **`somi.absorption`** | Absorb from other models | transplant, alignment, multi_model, integrity, probe fingerprint |
| **`somi.diagnostics`** | Health and levels | `DiagnosticDashboard`, Level 1–5 diagnostics, neuro |
| **`somi.compression`** | Compress brain | `AutoCompress` (auto trigger), `adaptive_compress`, stress/spectral pruning |
| **`somi.training`** | Training strategies | `DualLearningTrainer`, JEPA-style loss |
| **`somi.utils`** | Shared utilities | W&B logger, cost lock |

**Import pattern:**

```python
from somi import SOMIBrainConfig
from somi.brain.circuit_brain import SOMICircuitBrain
from somi.lm import SOMILanguageModel, create_somi_lm, Distiller, AutoGrowth
from somi.compression import AutoCompress
from somi.diagnostics.dashboard import DiagnosticDashboard
from somi.absorption.integrity import check_integrity
```

---

## 6. Experiments (What Each Script Does)

| Script | What it does |
|--------|----------------|
| **`experiments.demo`** | Circuit-S brain, one forward pass, full diagnostics, self-healing, brain scan. Best first run. |
| **`experiments.test_e2e`** | End-to-end: pure SOMI LM, growth, auto-growth, distillation, distillation+growth, generation. |
| **`experiments.test_growth`** | Single Part growth and behavior preservation. |
| **`experiments.simulate_lifecycle`** | Conception → gestation (grow + absorb) → birth (inference). |
| **`experiments.absorb_and_grow`** | Absorption from another model into a Circuit Brain; optional growth. |
| **`experiments.train_addition`** | Toy addition task; shows macro (backprop) + micro (physics) learning. |
| **`experiments.show_action_cascade`** | Shows how ALL parameters are derived from the action at every scale (S/M/L/XL). |

---

## 7. Important Classes and Functions

**Brain:**

- `SOMICircuitBrain(config, input_dim, output_dim)` — Full circuit. `forward(x, training=...)` returns `(logits, diagnostics)`.
- `brain.grow_brain()` — Grows all Parts and rewires white matter (used by AutoGrowth).

**LM:**

- `create_somi_lm(config, vocab_size, ...)` — Pure SOMI LM (no transformer).
- `SOMILanguageModel` — Wrapper interface.
- `Distiller(teacher, student, ...)` — Distill teacher logits into student.
- `AutoGrowth(model, stress_threshold, patience, ...)` — Grows neurons when stress is high and sustained.
- `AutoGrowthFull(brain, ...)` — Full structural growth: grows neurons, adds Parts (when shared Parts are saturated), adds Systems, and recalibrates physics after every growth event.

**Diagnostics:**

- `DiagnosticDashboard(config, auto_heal=..., wandb_log=...)` — Runs all diagnostics (L1–L5, neuro).
- `report = dashboard.report(brain, step=..., test_input=...)` — Returns health score, pathologies, fixes applied.

**Absorption:**

- See `somi.absorption`: transplant (delta, multi-model), alignment, integrity check, probe fingerprint.

**Config:**

- `SOMIBrainConfig.circuit_s()`, `.circuit_m()`, `.circuit_l()`, `.circuit_xl()` — Presets.
- `SOMIBrainConfig.auto(hidden_dim, n_parts)` — Zero-hyperparameter config (you choose size).
- `SOMIBrainConfig.from_scratch()` — True zero-input: starts H=16, P=1; grows from stress.

---

## 8. File Map (Where to Look)

- **Config and presets:** `somi/config.py`
- **9 forces, field equation:** `somi/physics/forces.py`
- **Settling (SSM, symplectic, spectral):** `somi/physics/settling.py`
- **Geometry update (stress → W):** `somi/physics/geometry.py`, `geometry_base.py`
- **Action-derived parameter cascade (L1-L5):** `somi/physics/action_derived.py`
- **Calibration (mass, beta, n_settle):** `somi/physics/calibration.py`
- **Part, System, WhiteMatter, CircuitBrain:** `somi/brain/part.py`, `system.py`, `white_matter.py`, `circuit_brain.py`
- **Pure SOMI LM, growth, distillation:** `somi/lm/model.py`, `growth.py`, `distill.py`
- **Diagnostics (all levels + neuro):** `somi/diagnostics/dashboard.py` and modules it uses
- **Absorption:** `somi/absorption/` (transplant, alignment, integrity, multi_model, etc.)
- **Training (dual learning, JEPA):** `somi/training/`

More detail: [docs/API_SOURCE_MAP.md](API_SOURCE_MAP.md), [docs/TARGET_ARCHITECTURE.md](TARGET_ARCHITECTURE.md).

---

## 9. Extending the Code

- **New force:** Add in `physics/forces.py` and wire into the combined force used in settling.
- **New diagnostic:** Add a check in `diagnostics/` and register it in the dashboard.
- **New experiment:** Add a script in `experiments/` that imports from `somi` and uses `SOMIBrainConfig` + `SOMICircuitBrain` (or LM, absorption, etc.). Follow existing experiments for W&B and cost lock if you train.
- **New preset:** Add a `@staticmethod` on `SOMIBrainConfig` (e.g. `circuit_xxl()`) in `config.py`.

---

## 10. Troubleshooting

- **Import errors:** Ensure you ran `pip install -e .` from the repo root and you’re running from that root (or with it on `PYTHONPATH`).
- **CUDA/device:** Code uses PyTorch; set `CUDA_VISIBLE_DEVICES` or use `.to(device)` as usual. Experiments that train should respect cost lock (see `somi.utils` and workspace rules).
- **Diagnostics report “pathologies”:** Normal on first steps or with random init. Dashboard can auto-heal some; for persistent issues, check geometry scale, learning rates, and calibration.
- **Growth not triggering:** AutoGrowth needs stress above threshold for several steps; increase stress or lower threshold for testing.
- **What’s implemented vs theory:** See [THEORY_AHEAD_OF_CODE.md](THEORY_AHEAD_OF_CODE.md) and [ABSORPTION_THEORY_VS_IMPLEMENTED.md](ABSORPTION_THEORY_VS_IMPLEMENTED.md).

---

## 11. Theory vs Code (Quick Ref)

| Need | Doc / code |
|------|------------|
| Understand SOMI conceptually | [USER_MANUAL_THEORY.md](USER_MANUAL_THEORY.md), [START_HERE_WHAT_IS_THIS.md](../START_HERE_WHAT_IS_THIS.md) |
| Install and run | This manual, README.md |
| What’s in each module | This manual §5, §7, §8; API_SOURCE_MAP.md |
| What theory exists but isn’t in code | THEORY_AHEAD_OF_CODE.md |
| Absorption: theory vs implemented | ABSORPTION_THEORY_VS_IMPLEMENTED.md |

---

*For the theory (5 levels, absorption, dual learning, symbols), see [USER_MANUAL_THEORY.md](USER_MANUAL_THEORY.md).*
