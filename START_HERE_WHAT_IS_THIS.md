# What Is This Folder? (Explained From Scratch)

You asked for an explanation of everything that was just created, as if you don't know what SOMI is. Here it is.

---

## What is SOMI?

**SOMI** stands for **Self-Organizing Models of Intelligence**.

In one sentence: **SOMI is a way to build AI that behaves more like a brain than like a traditional neural network.**

### The basic idea

Most AI today (including ChatGPT-style models) is built from **transformers**: layers that do a fixed sequence of math (attention, then a small “MLP” network). The connections between neurons are decided when you design the model and then trained with backpropagation. The structure doesn’t really “organize itself.”

SOMI is different. It is based on **physics-style equations** that describe:

1. **A field (φ, “phi”)** — Think of it like the “activity” or “thought” at each point in a network. It can change over time (it has a velocity, like a ball rolling).
2. **A geometry (W)** — A matrix that says “how connected” each part of the network is to each other part. This is not fixed; it **learns and changes** from stress and correlation (more like a brain wiring itself).
3. **Settling** — Instead of one shot through a layer, the activity **settles** over many small steps, like a ball rolling in a bowl until it comes to rest. Damping and forces (coupling, prediction error, etc.) determine where it ends up.

So: **SOMI = a dynamical system (field + geometry) that self-organizes.** The “physics” (forces, stress, geometry updates) is what makes it “SOMI,” not the fact that it has layers or parameters.

---

## Why does this folder exist?

You had SOMI code and theory spread across **several places**:

- `SOMI_Research` (SOMI_3_0, SOMI_4, implementations, experiments…)
- `SOMI Workspace` on Google Drive
- `_Distilled Theory of LLMs` (books/docs)
- Other learning vaults and archives

That made it messy: different folders imported from each other with path hacks, duplicate ideas, and no single “source of truth.”

**This folder (`somi`)** is a **clean rebuild**: one place that contains only what’s needed to **run** SOMI — no path hacks, no reaching into other workspaces. Everything you need to use SOMI as a library or run experiments is here.

---

## What’s actually in this folder?

### 1. The `somi` package (the “engine”)

This is the core code. Think of it as the SOMI “engine” you can import in Python.

| Folder / File | What it is in plain English |
|---------------|-----------------------------|
| **`somi/config.py`** | All the knobs for a SOMI brain: size, damping, learning rates, sparsity, etc. Includes an `auto()` mode that sets many of these from the size of the network so you don’t have to pick everything by hand. |
| **`somi/physics/`** | The “physics” of SOMI. |
| ↳ `core.py` | Laplacian (how activity spreads on the network), precision tracking, and the basal ganglia–style gate that filters which errors matter. |
| ↳ `forces.py` | The **9 forces** that act on the field φ (coupling, anchoring, saturation, prediction error, gating, error smoothing, coordination, damping, noise). This is the “field equation” in code. |
| ↳ `geometry.py` + `geometry_base.py` | How the **connectivity W** is updated from stress (the “geometry equation”), plus constraints (e.g. non-negative, sparse). |
| ↳ `settling.py` | **Settling**: symplectic (step-by-step), spectral (closed-form linear part), and **SSM** (fast closed-form + nonlinear correction). This is where “the ball rolls until it rests.” |
| ↳ `hamiltonian.py` | Energy of the system (for diagnostics and stability checks). |
| ↳ `calibration.py` | Mass, damping, and other parameters derived from the current W (so the physics stays in a good regime). |
| ↳ `neuromodulation.py` | Extra “neuromodulatory” state (e.g. arousal) that can modulate learning or dynamics. |
| **`somi/brain/`** | The “brain” built from that physics. |
| ↳ `part.py` | One **Part**: one region with its own W, mass, precision, and the ability to **grow** (add neurons). |
| ↳ `system.py` | A **System**: a pathway that routes data through certain Parts. |
| ↳ `white_matter.py` | **White matter**: long-range connections between Parts. |
| ↳ `circuit_brain.py` | The full **Circuit Brain**: multiple Parts + Systems + white matter, with a single `grow_brain()` that grows all Parts and rewires connections. |
| **`somi/lm/`** | Using SOMI as a **language model** (text in → next-word predictions). |
| ↳ `wrapper.py` | Wraps **any** HuggingFace language model (e.g. Llama, Mistral) so that each layer’s hidden state is passed through SOMI settling. You get “SOMI-enhanced Llama” etc. |
| ↳ `model.py` | A **pure SOMI language model**: no transformer inside. Just embedding → many SOMI-SSM layers (settling + geometry) → LM head. It can **grow** (add hidden units) when stressed. |
| ↳ `growth.py` | **Auto-growth**: watches stress; when it stays high for a while, it calls `grow_brain()` so the model gets more capacity. |
| ↳ `distill.py` | **Distillation**: train the pure SOMI model to match the output distributions of a “teacher” (e.g. a SOMI-wrapped Llama), so knowledge is transferred into the growing SOMI model. |
| **`somi/compression/`** | Ways to compress a SOMI brain (e.g. by stress or spectral importance) without breaking it. |
| **`somi/absorption/`** | Tools to **absorb** knowledge from another model: transplant weights, align dimensions, check integrity, multi-teacher distillation. |
| **`somi/diagnostics/`** | “Health check” for a SOMI brain: pathologies (e.g. exploding geometry), stability, energy, etc. |
| **`somi/visualization/`** | Plots and “brain scans” for activity, geometry, stress. |
| **`somi/training/`** | Training strategies: dual learning (macro + micro), test-time learning, JEPA-style prediction. |
| **`somi/utils/`** | Shared utilities: W&B logging, cost/safety limits so runs don’t blow your compute budget. |

So: **config + physics + brain + LM + compression + absorption + diagnostics + visualization + training + utils** = the whole SOMI “engine” in one package.

---

### 2. `experiments/` (scripts that *use* the engine)

These are **runnable scripts** that use the `somi` package. They’re not part of the installable library; they’re “things you can run to see SOMI in action.”

| File | What it does |
|------|------------------|
| **`demo.py`** | Builds a small Circuit Brain, runs one forward pass, runs diagnostics, and shows a simple “brain scan.” Good first run. |
| **`test_growth.py`** | Checks that a single Part can **grow** (add neurons) and that existing behavior is preserved. |
| **`simulate_lifecycle.py`** | Simulates “conception → gestation (grow + absorb) → birth (inference)” for a Circuit Brain. |
| **`absorb_and_grow.py`** | Uses the absorption tools to pull knowledge from another model into a Circuit Brain, and can grow the brain in the process. |
| **`train_addition.py`** | Trains a small Circuit Brain on a toy task (e.g. 2-digit addition) so you can see both “macro” (backprop) and “micro” (physics) learning. |
| **`test_e2e.py`** | **End-to-end test suite**: pure SOMI LM forward pass, growth, auto-growth, distillation, distillation+growth, and generation. If this passes, the clean rebuild is working. |

So: **experiments = demos and tests** that rely only on the `somi` package.

---

### 3. `theory/` and `docs/`

- **`theory/`** — Key reference documents copied in so you don’t have to leave this folder to read them: e.g. “SOMI from first principles,” “the 5 levels,” “growth solves local learning,” “SOMI life cycle,” symbol glossary.
- **`docs/`** — Notes created during the rebuild: **audit inventory** (what came from where), **API/source map** (which function lives in which file), **target architecture** (how the package is structured and what the rules are).

So: **theory = “what SOMI is” on paper; docs = “how this codebase was built and where things live.”**

---

### 4. Root files

- **`README.md`** — Short overview and how to install/run.
- **`pyproject.toml`** — Tells Python how to install the `somi` package (`pip install -e .`).
- **`MIGRATION_MAP.md`** — Old paths (e.g. in SOMI_Research or SOMI Workspace) → new paths in this folder. So you can trace “this file in the clean repo used to be that file over there.”

---

## How do I run something?

1. **Install the package once** (from this folder):
   ```bash
   pip install -e .
   ```
2. **Run experiments** (from this folder or with this folder on your path):
   ```bash
   python -m experiments.demo
   python -m experiments.test_e2e
   python -m experiments.train_addition
   ```
3. **Use SOMI in your own code**:
   ```python
   from somi import SOMIBrainConfig
   from somi.brain.circuit_brain import SOMICircuitBrain
   from somi.lm import SOMILanguageModel, create_somi_lm, Distiller, AutoGrowth
   ```

---

## One-sentence summary

**This folder is a single, clean copy of SOMI — the “self-organizing brain” engine plus its language-model use (wrapper, pure LM, growth, distillation) and experiments — with no dependency on other workspaces, so you can run and extend it from one place.**

If you want to go deeper next, we can open one file (e.g. `somi/physics/forces.py` or `somi/lm/model.py`) and walk through it line by line in the same “explain like I don’t know SOMI” style.
