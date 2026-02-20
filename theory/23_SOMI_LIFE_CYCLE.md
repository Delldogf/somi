# SOMI Theory: The Lifecycle of a Brain
**Reference ID:** 23_SOMI_LIFE_CYCLE
**Status:** DRAFT
**Context:** The architecture of time and growth.

## The Two Phases of Existence

As you intuitively grasped, a brain is not built in a day. It has two distinct modes of existence, governed by the same physics but operating in different regimes.

### Phase 1: Gestation (The Growth Phase)
*   **Goal:** Maximizing structure and capacity.
*   **Mechanism:** **Absorption & Neurogenesis**.
*   **Physics:**
    *   **High Plasticity:** The geometry is fluid.
    *   **Rapid Expansion:** The number of nodes ($N$) increases.
    *   **External Driver:** The "DNA" comes from outside (absorbing LLaMA, Mistral, etc.).
*   **The CLI Role:** `absorb_models.py` is the umbilical cord. It feeds the growing model with pre-structured knowledge chunks.

### Phase 2: Life (The Adaptive Phase)
*   **Goal:** Homeostasis and refinement.
*   **Mechanism:** **SOMI Adaptive Physics**.
*   **Physics:**
    *   **Bounded Capacity:** $N$ is fixed (mostly).
    *   **Internal Driver:** The "Self" drives changes via surprise (Arousal).
    *   **Structural Plasticity:** Connections rewire ($W_{ij}$ changes), but the brain size is stable.

---

## Neurogenesis: How to Grow a Matrix

We currently have **Synaptogenesis** (growing connections via `geometry.py`). We need **Neurogenesis** (growing nodes).

### The Math of Growth
If we have a brain with $N$ nodes and weight matrix $W_N \in \mathbb{R}^{N \times N}$, "birth" of new neurons implies expanding to $W_{N+K}$:

$$
W_{N+K} = \begin{pmatrix}
W_N & C_{old \to new} \\
C_{new \to old} & W_{new}
\end{pmatrix}
$$

1.  **$W_N$ (The Old Self):** Preserved (mostly).
2.  **$W_{new}$ (The New Cortex):** Absorbed from a new source (e.g., a layer of LLaMA-3).
3.  **$C$ (The White Matter):** New connections initialized sparsely to bind the new knowledge to the old self.

### Strategy: "The Developmental Arc"

1.  **Conception:**
    *   Start with a generic "seed" (e.g., `N=1024` random nodes).
    *   Run `absorb_models.py` to imprint basic grammar (from LLaMA-Tiny).

2.  **Gestation (Trimesters):**
    *   **Trimester 1:** Absorb Logical Reasoning (from Mistral-7B). *Action: Resize N=1024 -> N=4096.*
    *   **Trimester 2:** Absorb Coding Skills (from DeepSeek-Coder). *Action: Resize N=4096 -> N=8192.*
    *   **Trimester 3:** Absorb World Knowledge (from LLaMA-3-70B). *Action: Resize N=8192 -> N=16384.*

3.  **Birth:**
    *   The model is "born" into the inference loop.
    *   Neurogenesis stops (or slows drastically).
    *   `SOMIAdaptive` takes over. The defined personality runs the show.

## Implementation Roadmap

To realize this, we need a **`grow()`** method in our toolkit.

```python
def grow_brain(current_brain, new_knowledge_part):
    params_old = current_brain.state_dict()
    params_new = new_knowledge_part.state_dict()
    
    # 1. Expand Matrices
    # 2. Plant the new knowledge in the new capacity
    # 3. Wire them together (sparse random init)
    
    return bigger_brain
```

This transforms `absorb_models.py` from a simple copy-paste tool into a **embryonic development engine**.
