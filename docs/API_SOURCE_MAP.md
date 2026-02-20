# SOMI API Source Map

This map records provenance of key functions/classes in the clean package.

## Physics

- `compute_laplacian()` -> `somi/physics/core.py` (from `implementations/somi_2_0/core.py`)
- `PrecisionTracker` -> `somi/physics/core.py` (from `implementations/somi_2_0/core.py`)
- `BasalGangliaGate` -> `somi/physics/core.py` (from `implementations/somi_2_0/core.py`)
- `compute_field_forces()` -> `somi/physics/forces.py` (SOMI_4 enhanced 9-force version)
- `compute_stress_tensor()` -> `somi/physics/geometry.py` (SOMI_4 wrapper over base stress)
- `initialize_W()` -> `somi/physics/geometry_base.py` (from `implementations/somi_2_0/geometry.py`)
- `structural_plasticity()` -> `somi/physics/geometry_base.py` (from `implementations/somi_2_0/geometry.py`)
- `mass_conductivity_constraint()` -> `somi/physics/geometry.py` (SOMI_4 addition)
- `settle_symplectic()` -> `somi/physics/settling.py`
- `settle_spectral()` -> `somi/physics/settling.py`
- `settle_ssm()` -> `somi/physics/settling.py` (clean rebuild phase)
- `HamiltonianTracker` -> `somi/physics/hamiltonian.py`
- calibration functions (`compute_mass_vector`, `compute_beta`, `compute_n_settle`, etc.) -> `somi/physics/calibration.py`
- neuromodulation (`NeuromodulatorState`, `NeuromodulatorSystem`) -> `somi/physics/neuromodulation.py`

## Brain

- `SOMIPart` -> `somi/brain/part.py` (includes `grow()` neurogenesis)
- `SOMISystem` -> `somi/brain/system.py`
- `WhiteMatterTract` -> `somi/brain/white_matter.py`
- `SOMICircuitBrain` -> `somi/brain/circuit_brain.py` (includes `grow_brain()`)

## Language Model

- `create_somi_lm()` -> `somi/lm/wrapper.py`
- `SOMISSMLayer` -> `somi/lm/model.py`
- `SOMILanguageModel` -> `somi/lm/model.py`
- `AutoGrowth` -> `somi/lm/growth.py`
- `Distiller` -> `somi/lm/distill.py`

## Compression / Absorption / Diagnostics / Visualization / Training

- Directly sourced from SOMI_4 modules into corresponding `somi/*` packages with import cleanup.

## Utilities

- `SOMIWandbLogger` and W&B helpers -> `somi/utils/wandb_logger.py`
- `enforce_cost_lock`, `CostTimer`, `safe_wandb_init`, `safe_wandb_finish` -> `somi/utils/cost_lock.py`
