# Migration Map

This file maps old paths to new canonical paths.

## Core mapping

- `SOMI_Research/SOMI_4/config.py` -> `somi/config.py`
- `SOMI_Research/SOMI_4/physics/*` -> `somi/physics/*`
- `SOMI_Research/SOMI_4/brain/*` -> `somi/brain/*`
- `SOMI_Research/SOMI_4/compression/*` -> `somi/compression/*`
- `SOMI_Research/SOMI_4/absorption/*` -> `somi/absorption/*`
- `SOMI_Research/SOMI_4/diagnostics/*` -> `somi/diagnostics/*`
- `SOMI_Research/SOMI_4/visualization/*` -> `somi/visualization/*`
- `SOMI_Research/SOMI_4/training/*` -> `somi/training/*`
- `SOMI_Research/SOMI_4/lm/somi_wrapper.py` -> `somi/lm/wrapper.py`
- `SOMI_Research/SOMI_4/lm/somi_ssm_lm.py` -> `somi/lm/model.py`
- `SOMI_Research/SOMI_4/lm/auto_growth.py` -> `somi/lm/growth.py`
- `SOMI_Research/SOMI_4/lm/distill.py` -> `somi/lm/distill.py`

## Absorbed foundation mapping

- `SOMI_Research/implementations/somi_2_0/core.py` -> `somi/physics/core.py`
- `SOMI_Research/implementations/somi_2_0/geometry.py` -> `somi/physics/geometry_base.py`
- `SOMI_Research/implementations/somi_2_0/calibration.py` -> `somi/physics/calibration.py`
- `SOMI_Research/implementations/somi_2_0/neuromodulation.py` -> `somi/physics/neuromodulation.py`
- `SOMI_Research/implementations/somi_2_0/cost_lock.py` -> `somi/utils/cost_lock.py`

## Utilities and experiments mapping

- `SOMI_Research/utils/wandb_logger.py` -> `somi/utils/wandb_logger.py`
- `SOMI_Research/SOMI_4/demo.py` -> `experiments/demo.py`
- `SOMI_Research/SOMI_4/experiments/*` -> `experiments/*`

## Theories and references

- Key theory docs copied into `theory/` for reference.
- Full historical docs remain in original locations as read-only archives.
