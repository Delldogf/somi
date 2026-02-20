# SOMI Audit Inventory

This inventory classifies SOMI assets across all discovered locations into:
- `keep-core`
- `keep-reference`
- `replace`
- `archive`

## Source Locations

1. `c:/Users/johna/Documents/SOMI_Research/`
2. `g:/My Drive/SOMI Workspace/`
3. `G:/My Drive/_Distilled Theory of LLMs/`
4. `c:/Users/johna/OneDrive/Documents/_SOMI/`
5. `c:/Users/johna/Documents/SOMI_Learning_Vault_2026_02_16/`
6. `c:/Users/johna/OneDrive/Desktop/for chatgpt/`
7. `c:/Users/johna/OneDrive/Documents/Winning AIMO 3/`

## Classification

## keep-core

- `SOMI_Research/SOMI_4/config.py`
- `SOMI_Research/SOMI_4/physics/*.py`
- `SOMI_Research/SOMI_4/brain/*.py`
- `SOMI_Research/SOMI_4/compression/*.py`
- `SOMI_Research/SOMI_4/absorption/*.py`
- `SOMI_Research/SOMI_4/diagnostics/*.py`
- `SOMI_Research/SOMI_4/visualization/*.py`
- `SOMI_Research/SOMI_4/training/*.py`
- `SOMI_Research/SOMI_4/experiments/{test_growth.py,simulate_lifecycle.py,absorb_and_grow.py,train_addition.py}`
- `SOMI_Research/SOMI_4/demo.py`
- `SOMI_Research/SOMI_4/lm/{somi_wrapper.py,somi_ssm_lm.py,auto_growth.py,distill.py,test_e2e.py}`
- `SOMI_Research/implementations/somi_2_0/{core.py,geometry.py,calibration.py,neuromodulation.py,cost_lock.py}`
- `SOMI_Research/utils/wandb_logger.py`

## keep-reference

- `SOMI_Research/SOMI_3_0/` theory and status docs
- `SOMI_Research/SOMI_Master/` master documentation
- `G:/My Drive/_Distilled Theory of LLMs/SOMI_FOR_DUMMIES_SERIES/` educational content
- `OneDrive/Documents/_SOMI/` mirrored educational docs
- `Documents/SOMI_Learning_Vault_2026_02_16/` learning vault

## replace

- `SOMI_Research/implementations/somi_lm/somi_language_model.py` (superseded by clean `somi/lm/wrapper.py`)
- `SOMI_Research/implementations/core/*.py` old core variants superseded by merged SOMI_4 + SSM LM path
- `G:/My Drive/SOMI Workspace/implementations_somi_2_0/*` superseded by curated `somi/physics/*`

## archive

- `SOMI_Research/wandb/` run logs
- `SOMI_Research/archive/` historical files
- `OneDrive/Desktop/for chatgpt/_Distilled Theory of LLMs/archive/` historical mirrors
- `SOMI_Research/SOMI_Distill/` separate web app project (not part of clean runtime package)

## Output of this rebuild

Canonical runtime code now lives in:
- `c:/Users/johna/Documents/somi/somi/`
