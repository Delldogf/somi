# SOMI Clean Rebuild

This is the clean standalone SOMI package rebuilt from validated sources.

## Install

```bash
pip install -e .
```

## Package layout

- `somi.physics`: forces, geometry, settling, hamiltonian, calibration, neuromodulation
- `somi.brain`: part, system, white matter, circuit brain
- `somi.lm`: wrapper, pure SOMI LM, auto-growth, distillation
- `somi.compression`, `somi.absorption`, `somi.diagnostics`, `somi.visualization`, `somi.training`
- `somi.utils`: utilities including W&B logger and cost lock

## Run tests and demos

```bash
python -m experiments.test_e2e
python -m experiments.test_growth
python -m experiments.simulate_lifecycle
python -m experiments.absorb_and_grow
python -m experiments.train_addition
```
