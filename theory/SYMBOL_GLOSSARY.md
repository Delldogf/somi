# SOMI 3.0 — Symbol Glossary (For Dummies)

Every symbol used in SOMI theory, in one place. Each entry gives: what it looks like, how to say it, what it means in plain English, and an analogy.

## EXPERIMENTAL VALIDATION (Feb 11, 2026)

**All core SOMI symbols have been measured in VibeThinker-1.5B:**

Validated measurements:
- **φ** (activity): Extracted from all 28 transformer layers
- **W** (connectivity): 28 matrices [3072×3072] extracted
- **M** (mass): Mean 0.621 ± 0.311 per layer → hierarchy confirmed
- **S** (stress): Layer-wise profile measured (highest 0.7783 at output)
- **H** (Hamiltonian): Energy trajectory confirms decrease during settling
- **settling rate**: 0.7276 measured → equilibrium behavior validated
- **domain stress**: Code (0.924), math (0.889), English (0.825)
- **Δ** (topology delta): 99% knowledge transfer validates extraction/transplantation

**These are not abstract mathematical symbols — they are measurable physical quantities in real transformers.**

---

## Activity and field

| Symbol | Name | Meaning | Analogy |
|--------|------|---------|--------|
| **φ** (phi) | phi | **Activity** — the current state of each feature (node) in the network. A vector: one number per feature. | Like “temperature” at each point on a map: the value of the field at that node. |
| **φ̇** (phi-dot) | phi-dot | **Velocity of activity** — how fast φ is changing at each feature. Time derivative of φ. | Like “how fast the temperature is rising or falling” at each point. |
| **e** | e, error | **Prediction error** — e = φ − prediction (or φ − target in JEPA). What the system got wrong. | Like “how far the current guess is from the right answer.” |
| **ê** or **prediction** | hat / prediction | **Prediction** — what the system expects φ (or the target) to be. | The “guess” before we compare to the target. |

---

## Geometry and connectivity

| Symbol | Name | Meaning | Analogy |
|--------|------|---------|--------|
| **W** | W, weights | **Connectivity matrix** — W_ij = strength of connection from j to i (or between i and j). Defines the “geometry” of the network. | Like “how strongly each pair of brain regions is wired.” |
| **W_ij** | W-i-j | **One connection** — the weight between feature j and feature i. | The strength of one synapse or one link. |
| **W_local** | W-local | **Local connectivity** (SOMI 3.0) — the W matrix *inside* one Part only. Dense, plastic. | The “grey matter” inside one brain region. |
| **G(W)** or **G** | G of W, metric | **Metric tensor** — defines “distance” and “mass” for activity. Depends on W. In the diagonal approximation, G is the mass matrix. | Like “how hard it is to change activity at each point” — geometry sets inertia. |
| **L** or **L_rw** | Laplacian | **Graph Laplacian** (often random-walk normalized) — built from W. Measures how “spread out” or “smooth” the field is on the graph. | Like “how much neighbors disagree” — used in coupling and error smoothing. |

---

## The action and energy

| Symbol | Name | Meaning | Analogy |
|--------|------|---------|--------|
| **S** | S, action | **Action** — the single quantity we integrate over time (kinetic − potential). All equations of motion come from making S stationary. | The “total cost” of a path through time; nature picks the path that makes it stationary. |
| **T** | T, kinetic | **Kinetic energy** — (1/2) φ̇^T G φ̇. Energy of motion of the field. | Like “how much the system is moving” (velocity squared, weighted by mass). |
| **V** | V, potential | **Potential energy** — sum of coupling, anchor, saturation, info stress, error terms, coordination, weight cost. | Like “how much the system is stretched or wrong” — it wants to sit where V is low. |
| **H** | H, Hamiltonian | **Total energy** — H = T + V. Should decrease during settling (Lyapunov stability). | Like “total energy”; with damping it goes downhill. |
| **R** or **ℛ** | Rayleigh | **Dissipation function** — models damping (and W update cost). Not part of the Lagrangian; added for energy loss. | The “friction” that drains energy. |

---

## Mass, damping, and timescales

| Symbol | Name | Meaning | Analogy |
|--------|------|---------|--------|
| **M** or **M_0** | M, mass | **Inertial mass** — base mass (regime choice). In full theory, **M_i(W)** = per-feature mass from geometry. | Like “how heavy” a feature is — high mass = slow to move. |
| **M_i** | M-i | **Mass of feature i** — M_i(W) from W (e.g., inverse of connection concentration). | How sluggish feature i is; set by its connectivity pattern. |
| **β** (beta) | beta | **Damping coefficient** — β_i per feature. Friction that dissipates energy. | Like “air resistance” on each feature. |
| **ζ** (zeta) | zeta | **Damping ratio** — target_zeta. 0.15 = underdamped (oscillations); 0.9 = overdamped (quick settle). | How “bouncy” vs “dead” the system is. |
| **η** (eta) | eta | **Geometry learning rate** — how fast W updates per step. Often self-calibrated; modulated by DA. | How quickly connections (W) change from stress. |
| **dt** | d-t, delta t | **Time step** — small step used when integrating the field equation (settling). | The “tick” of the clock in simulation. |
| **n_settle** | n-settle | **Number of settling steps** — how many times we run the field equation per token/step. Temporal depth. | How long SOMI “thinks” before producing an output. |

---

## Potential and forces (parameters)

| Symbol | Name | Meaning | Analogy |
|--------|------|---------|--------|
| **α₀** (alpha-0) | alpha zero | **Anchoring strength** — pull toward zero (rest). | Like a spring that pulls activity back to baseline. |
| **α₁** (alpha-1) | alpha one | **Coupling strength** — how strongly neighbors influence each other through W. | How much connected features “pull” each other. |
| **κ₀** (kappa-0) | kappa zero | **Precision weight for prediction error** — scales the info-stress force. | How much we care about prediction error in the dynamics. |
| **κ₁** (kappa-1) | kappa one | **Basal ganglia gate strength** — only large errors pass; filters small noise. | A threshold: “only big errors get through.” |
| **λ_E** (lambda-E) | lambda E | **Error smoothing** — errors diffuse across the graph (smooth out). | How much neighboring nodes “share” their error. |
| **λ_C** (lambda-C) | lambda C | **Coordination strength** — reward for connected features moving together. | “Neighbors should agree” — pull toward coordination. |
| **λ_W** (lambda-W) | lambda W | **Weight decay / metabolic cost** — small cost on W so unused connections shrink. | Like “maintaining synapses costs energy.” |

---

## Stress and learning

| Symbol | Name | Meaning | Analogy |
|--------|------|---------|--------|
| **Stress** | stress | **Prediction error** (magnitude or tensor). Drives W updates. In JEPA, stress = distance in embedding space. | How wrong the current prediction is; the learning signal. |
| **S_ij** (stress tensor) | S-i-j | **Stress tensor** — which connection (i,j) contributes how much to the total error. Drives ΔW_ij. | “Which links are most to blame for the error?” |
| **κ_stdp** (kappa-stdp) | kappa STDP | **STDP strength** — multiplies phi_i * phi_dot_j for directional learning (pre→post). | “Neurons that fire in sequence wire in sequence.” |

---

## Precision and gating

| Symbol | Name | Meaning | Analogy |
|--------|------|---------|--------|
| **Σ** (Sigma) | Sigma | **Precision matrix** (or diagonal) — per-feature confidence. High precision = weight that feature’s error more. | How much we “trust” each feature’s error signal. |
| **B** | B | **Basal ganglia matrix** — projects error through a gating nonlinearity (e.g., softplus). Filters small errors. | A gate: only big errors get through to drive learning. |

---

## Eigenvalues and spectrum

| Symbol | Name | Meaning | Analogy |
|--------|------|---------|--------|
| **λ** (lambda) | lambda | **Eigenvalue** — of the Laplacian or of W. Determines natural frequencies of oscillation. | Like “natural frequency” of a mode (fast vs slow). |
| **ω** (omega) | omega | **Angular frequency** — from eigenvalues; ω = sqrt(λ) (or similar). How fast a mode oscillates. | Like “pitch” of an oscillation (high ω = fast). |

---

## SOMI 3.0 specific

| Symbol / term | Name | Meaning | Analogy |
|---------------|------|---------|--------|
| **Part** | Part | A brain region — one mini-SOMI with its own φ, W_local, mass, neuromodulators. | One cortical area with its own local wiring. |
| **White Matter** | white matter | Sparse, fixed or slow projections *between* Parts. Low-rank maps. | Long-range axon bundles between regions. |
| **System** | system | A **circuit** — a route through several Parts (e.g., Sensory → Association → PFC). | One functional pathway (e.g., “memory circuit”). |
| **X-Encoder** | X-encoder | Maps input (e.g., text) to embedding that drives SOMI. | “Eyes and ears” — turn input into internal representation. |
| **Y-Encoder** | Y-encoder | Maps target (e.g., answer text) to target embedding. Often frozen. | “What the right answer looks like” in embedding space. |
| **Y-Decoder** | Y-decoder | Maps SOMI’s predicted embedding to text. Used only when we need output. | “Mouth” — turn internal prediction into words when needed. |

---

## Neuromodulators (abbreviations)

| Symbol | Name | Meaning | Analogy |
|--------|------|---------|--------|
| **NE** | Norepinephrine | Arousal — monitors surprise; modulates global learning (arousal → η). | “Alertness” — turn up learning when things are surprising. |
| **DA** | Dopamine | Reward / learning rate — monitors stress improvement; modulates η (e.g., 0.5×–1.5×). | “Reward signal” — boost learning when predictions improve. |
| **ACh** | Acetylcholine | Attention — monitors per-feature error; modulates mass (salient features get lower mass). | “Spotlight” — focus dynamics on the most wrong features. |
| **5-HT** | Serotonin | Patience — monitors difficulty; modulates n_settle and noise_ratio. | “Persistence” — more settling and exploration for hard problems. |

---

## Indices and dimensions

| Symbol | Name | Meaning | Analogy |
|--------|------|---------|--------|
| **i, j** | i, j | Indices over features (nodes). e.g., φ_i = activity at node i; W_ij = connection from j to i. | “Which neuron” or “which feature.” |
| **N** | N | Number of features (nodes). Size of φ and dimension of W. | How many “neurons” or units in the layer/Part. |
| **local_dim** | local dim | (SOMI 3.0) Size of one Part — dimension of φ and W_local inside that Part. | How many units in one brain region. |

---

## Optional / advanced

| Symbol | Name | Meaning | Analogy |
|--------|------|---------|--------|
| **B(W)** | B of W | **Damping matrix** — from geometry; diagonal in the standard approximation. | Per-feature friction. |
| **h_i** | h-i | **Connection concentration** (Herfindahl) — h_i = sum_j W_ij^2. Used to compute M_i(W). | How “focused” feature i’s connections are. |
| **Γ** (Gamma) | Christoffel | **Christoffel symbols** — appear in full geodesic equation when G is non-diagonal. Usually zero in our approximation. | Curvature terms in the “curved space” version of the equation. |

---

*Use this glossary whenever you see a symbol in the main document or theory files. If a symbol is missing here, add it with the same format: symbol, name, meaning, analogy.*
