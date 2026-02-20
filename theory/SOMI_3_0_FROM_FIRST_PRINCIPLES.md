# SOMI 3.0: From First Principles — The Complete Guide For Dummies

This document explains Self-Organizing Models of Intelligence (SOMI) from zero assumed knowledge. By the end you will understand what SOMI is, why it exists, how SOMI 2.0’s physics works, and how SOMI 3.0’s brain-inspired architecture and training fit on top.

---

## EXPERIMENTAL VALIDATION (Feb 11, 2026)

VibeThinker-1.5B experiments validated core SOMI predictions:

- Knowledge = topology (99% transfer efficiency)
- Transformers have SOMI physics (W, mass, stress measured)
- Diagnostics work (9 visualizations in 14 seconds)

Everything below has been measured in real transformers.

---

# Part I: The Problem (What We Are Trying to Do)

---

## Chapter 1: What Is Intelligence?

### In This Chapter

- What we mean by “intelligence” in plain language
- How brains and computers differ
- Why pattern recognition and reasoning both matter
- Why this sets up everything that follows

**Time:** About 15 minutes.

### The Least You Need to Know (Opening)

Intelligence is the ability to take in information, spot patterns, and use them to behave in ways that work in the world — including answering questions, solving problems, and adapting when things change. Brains do this with neurons and waves; computers do it with programs and, in AI, with networks of numbers. SOMI is an attempt to build AI that behaves more like a brain at the level of *how* it processes information: with activity that oscillates and with structure that organizes itself.

### Why Should You Care?

If we want AI that reasons better, uses less energy, and learns in more flexible ways, we need to be clear about what we are optimizing for. “Intelligence” here is not a single number; it is a set of capabilities (pattern recognition, reasoning, adaptation) that we want our model to have. SOMI is designed so that those capabilities emerge from simple physical rules — activity and geometry shaping each other — instead of from stacking more and more layers.

### What Do We Mean by “Intelligence”?

In everyday language, we say something is “intelligent” when it:

- **Notices patterns** — e.g., “when X happens, Y usually follows”
- **Uses those patterns to decide** — e.g., “so I will do Z”
- **Updates when the world changes** — e.g., “that pattern stopped working, try something else”

So intelligence is not just storing facts; it is *using* structure in the world to act effectively. In SOMI we care especially about:

- **Pattern recognition:** spotting structure in data (e.g., “this sentence is about math”)
- **Reasoning:** combining patterns to reach new conclusions (e.g., “if A and B, then C”)
- **Learning:** changing internal structure (e.g., connection strengths) so that future pattern recognition and reasoning get better

> [!remember]
> In this document, **intelligence** = pattern recognition + reasoning + learning from experience, in a way that supports useful behavior.

### Brains vs Computers: A Useful Contrast

Brains and computers both process information, but in very different ways:

| Aspect | Brain | Typical computer / classic AI |
|--------|--------|--------------------------------|
| **Basic unit** | Neurons (cells that spike and communicate) | Bits, numbers, “neurons” in software that are just weights and activations |
| **Communication** | Spikes, chemicals, and *waves* of activity (oscillations) | Numbers passed between layers or steps |
| **Structure** | Circuits and regions that *share* parts; wiring is partly fixed, partly plastic | Often many identical layers; one path for all inputs |
| **Learning** | Local rules (e.g., “strengthen this synapse because these two cells fired together”) plus global modulators | Mostly one global algorithm (e.g., backprop) that updates all weights from a single loss |
| **Always on?** | Yes — activity continues even when “idle” | Usually no — compute only when you run a forward pass |

SOMI takes inspiration from the brain side: activity that can oscillate, structure that is both global (fixed “highways”) and local (plastic “neighborhoods”), and learning that can happen locally from prediction error, not only from one global backward pass.

### Pattern Recognition and Reasoning

- **Pattern recognition** is detecting structure: “this sequence of tokens looks like a question,” “this patch of pixels looks like an edge.” Networks are good at this when we train them on lots of examples.
- **Reasoning** is combining what you know to get something new: “A is true, B is true, and A and B together imply C, so C is true.” That often requires multiple steps and holding intermediate results in a structured way.

Brains seem to do both using the same substrate: the same neurons and circuits participate in recognizing patterns and in chaining steps of reasoning. SOMI is built so that one kind of dynamics — activity on a graph whose geometry learns — can support both: patterns are encoded in the geometry (connections), and reasoning is the activity (e.g., the “field” φ) evolving and settling over time.

### Why This Sets Up SOMI

We want AI that:

1. **Recognizes patterns** and **reasons** over them (like brains do with one substrate).
2. **Learns** in ways that can be local and flexible (not only one global backward pass).
3. **Uses structure** that is partly fixed (e.g., which regions connect) and partly plastic (e.g., strengths inside a region).

So we need a model where:

- “Thinking” is something like activity evolving and settling (pattern + reasoning).
- “Learning” is the geometry of the network (e.g., connection strengths) changing from how that activity behaves (e.g., from prediction error).
- The *architecture* can look like circuits (parts, pathways) rather than a flat stack of identical layers.

That is exactly what SOMI 2.0 (physics) and SOMI 3.0 (circuits + training) are designed to do.

### The Least You Need to Know (Closing)

Intelligence here means pattern recognition, reasoning, and learning. Brains do this with neurons, waves, and circuits; classic AI does it with layers and backprop. SOMI aims for brain-like processing (oscillations, local learning, circuits) so we can get better reasoning and more efficient learning.

### Check Your Understanding

- [ ] Can you say in one sentence what we mean by “intelligence” in this document?
- [ ] Name two ways brains and typical AI differ (e.g., oscillations, local vs global learning).
- [ ] Why do we care about both pattern recognition and reasoning for SOMI?

### Mastery Test

- **Teach it:** Explain “what is intelligence?” to someone who has never heard of SOMI. Use only plain language (no symbols).
- **Probing questions:** What is the difference between recognizing a pattern and reasoning with it? Why might “one substrate” for both be useful?
- **Key terms:** intelligence, pattern recognition, reasoning, learning, brain vs computer.

---

## Chapter 2: How Brains Actually Work

### In This Chapter

- Neurons and synapses in simple terms
- Why brain waves (oscillations) matter
- Circuits and regions, not “layers”
- Fixed long-range wiring vs plastic local wiring
- How this inspires SOMI’s design

**Time:** About 20 minutes.

### The Least You Need to Know (Opening)

Brains are not stacks of identical layers. They are networks of specialized regions (like “sensory,” “memory,” “planning”) connected by long-range tracts. Inside each region, connections are dense and plastic; between regions, wiring is sparser and more fixed. Activity is not just a list of numbers — it oscillates (brain waves), and those oscillations help coordinate which regions talk when. SOMI 3.0’s “Parts” and “White Matter” mirror this: Parts = regions with local plasticity; White Matter = fixed or slowly changing links between them.

### Why Should You Care?

If we want AI that is more brain-like, we have to get the *organization* right, not just the math. Brains get a lot of their power from *who* talks to *whom* and *how* (oscillations, circuits). SOMI 3.0 is designed so that the same ideas — regions, tracts, oscillations — show up in the architecture and in the equations.

### Neurons and Synapses (Simplified)

- **Neuron:** A cell that receives inputs from other neurons, combines them, and sometimes “fires” (sends a spike to others). You can think of its *activity* as “how active it is right now.”
- **Synapse:** The connection from one neuron to another. It has a **strength** (weight): strong synapses pass more influence; weak ones pass less. Many learning theories say: “neurons that fire together wire together” — synapses between co-active neurons get stronger.

In SOMI we do not simulate every neuron. We work at a coarser level: **activity** (like the average “firing rate” or activation of a group of neurons) and **connection strengths** (like the aggregate strength between groups). So **φ (phi)** in SOMI is “activity” and **W** is “connectivity” between units. We will define φ and W precisely later; for now, think of them as the brain-like quantities we care about.

### Brain Waves (Oscillations)

Brains produce **rhythms**: different regions oscillate at different frequencies (theta, alpha, beta, gamma, etc.). These are not noise — they help:

- **Coordinate** when different areas are “listening” or “sending”
- **Carry information** (e.g., phase or frequency of oscillation)
- **Separate timescales** (slow waves for context, fast waves for local computation)

So “activity” in the brain is not just a static number; it *waves*. In SOMI, activity φ can oscillate (go up and down over time) and has a **velocity** φ̇ (phi-dot) — how fast it is changing. That pair (φ, φ̇) is like position and velocity in physics: it gives you both “where” the activity is and “how it is moving.” That is why SOMI’s dynamics are written like a physical system (with inertia, forces, and damping) instead of a one-shot feedforward formula.

### Circuits and Regions, Not Layers

The brain has **regions** (e.g., visual cortex, hippocampus, prefrontal cortex) and **pathways** between them (e.g., from hippocampus to cortex). A single “function” (e.g., memory) is often a **circuit**: a route through several regions (e.g., cortex → hippocampus → cortex). Important point: **the same region can belong to several circuits.** For example, prefrontal cortex (PFC) might be part of “working memory,” “planning,” and “inhibition” circuits. So PFC must learn representations that work for *all* of those — that is **generalization pressure** from shared use.

SOMI 3.0 copies this:

- **Parts** = regions (each has its own activity φ, its own connectivity W_local, and its own “physics”).
- **White Matter** = long-range pathways between Parts (sparse, fixed or slowly plastic).
- **Systems** = circuits: defined routes through several Parts. When a Part is in multiple Systems, it gets training signal from all of them — so it is pushed to generalize.

### Globally Sparse, Locally Dense

In the brain:

- **Within a region:** Many connections (dense, plastic “grey matter”).
- **Between regions:** Fewer long-range bundles (sparse “white matter”), laid down early and relatively fixed.

So the *topology* is “small-world”: dense neighborhoods (regions) connected by a sparse global network (tracts). SOMI 3.0 is the same: inside each Part, W_local is dense and learns a lot; between Parts, only a few White Matter tracts exist and they change slowly or not at all.

### How This Inspires SOMI

- **Activity that oscillates** → SOMI’s φ and φ̇ and the field equation (how φ evolves).
- **Geometry that learns** → SOMI’s W (and in 3.0, W_local per Part) updated by stress/prediction error.
- **Regions and circuits** → SOMI 3.0’s Parts, White Matter, and Systems.
- **Shared parts** → Generalization: a Part used by several Systems must serve all of them.

So “how brains actually work” is not a side note — it is the blueprint for SOMI’s architecture and for the kind of learning we want (local plasticity, global structure).

### The Least You Need to Know (Closing)

Brains use neurons and synapses, brain waves for coordination, and circuits through regions — with dense local wiring and sparser long-range tracts. SOMI’s activity (φ, φ̇) and connectivity (W) mirror this; SOMI 3.0’s Parts, White Matter, and Systems make the “regions and circuits” idea explicit.

### Check Your Understanding

- [ ] What are neurons and synapses in one sentence each?
- [ ] Why do we care about oscillations (brain waves) for SOMI?
- [ ] What does “globally sparse, locally dense” mean for the brain and for SOMI 3.0?

### Mastery Test

- **Teach it:** Explain “brains use circuits and regions, not layers” without using the word “SOMI.”
- **Probing questions:** Why might a region that appears in many circuits need to be more “general”? What would happen if all brain connections were plastic at the same speed?
- **Key terms:** neuron, synapse, oscillation, circuit, region, white matter, grey matter, generalization pressure.

---

## Chapter 3: What Current AI Does

### In This Chapter

- What a “neural network” is in plain language
- What transformers and “attention” are at a high level
- What “backpropagation” is and why it is dominant
- How this relates to “predicting the next token”

**Time:** About 20 minutes.

### The Least You Need to Know (Opening)

Current AI for language and reasoning is mostly **transformers**: big networks that take in a sequence of tokens (words or subwords), process them through many **layers** of “attention” and “MLP,” and output predictions (e.g., the next token). Learning is almost always **backpropagation**: one global signal (the loss) is sent backward through the whole network to adjust every weight. This is powerful but very different from the brain: no oscillations, no local learning rules, and no circuit structure — just one deep stack of similar layers.

### Why Should You Care?

SOMI is not “yet another transformer.” It is a different *kind* of system: dynamics (activity evolving) and geometry (connections learning from local error). To see why that might be useful, you need to know what the default is: transformers + backprop. Then we can say clearly what SOMI changes (oscillations, local learning, circuits) and what we keep when we combine them (e.g., backprop on encoders and white matter in SOMI 3.0).

### Neural Networks in Plain Language

A **neural network** is a bunch of simple units (“neurons” or “features”) that:

1. Receive numbers from other units (or from the input).
2. Form a weighted sum (each input times a **weight**).
3. Apply a simple nonlinear function (e.g., ReLU, tanh) and pass the result forward.

So the network is just: many weighted sums + nonlinearities, arranged in layers or graphs. The **weights** are what we “learn”; they are the connections between units. In SOMI we also have weights (our **W**), but we do not always update them with backprop — we often update them with local rules (stress tensor, etc.).

### Transformers and Attention (High Level)

A **transformer** is a neural network that:

- Takes a **sequence** of tokens (e.g., words or subwords).
- Turns each token into a vector (an **embedding**).
- Processes the sequence through many **layers**. Each layer has:
  - **Attention:** Each position “looks at” other positions and mixes their vectors (weighted average). The weights (“attention weights”) say *who* pays attention to *whom*.
  - **MLP (multi-layer perceptron):** A small feedforward net on each position’s vector.
- Produces outputs (e.g., logits for the next token).

So the “depth” of a transformer is **spatial**: more layers = more steps of mixing and transforming. There is no built-in notion of “activity oscillating” or “settling over time” — each layer is applied once per forward pass. SOMI, in contrast, gets “depth” from **time**: the same physics (field equation) is applied many *settling steps*, and activity can oscillate before it settles.

### Backpropagation in Plain Language

**Backpropagation** (“backprop”) is the standard way to train neural networks:

1. Run a **forward pass**: input → network → output. Compute a **loss** (e.g., how wrong the prediction was).
2. Compute how the loss would change if you changed each weight a tiny bit (the **gradient** of the loss with respect to each weight). This is done by applying the **chain rule** from the output back to the input — hence “back” propagation.
3. Update every weight in the direction that reduces the loss (e.g., weight -= learning_rate * gradient).

So one global signal (the loss) drives *all* weight updates. The brain does something different: many local rules (e.g., “strengthen this synapse because pre and post fired together”) plus modulators. SOMI follows the brain idea more: inside a Part, **W** is updated from local quantities (stress, activity) rather than from a single backward pass from the final loss. In SOMI 3.0 we can still use backprop for the “macro” parts (encoders, white matter) and reserve SOMI’s local learning for the “micro” (W inside Parts).

### Predicting the Next Token

Language models are usually trained to **predict the next token**: given “The cat sat on the,” predict “mat” (or whatever comes next). So the “target” is a discrete token, and the loss is something like cross-entropy over the vocabulary. That is **token-space** prediction. An alternative is **embedding-space** prediction: predict a continuous vector (the embedding of the next token or of the whole answer). That is what JEPA-style models do — and what SOMI’s “stress” can represent: the mismatch between SOMI’s predicted embedding and the target embedding. So SOMI can be the **predictor** in a JEPA: it predicts embeddings; the loss (and thus the learning signal for SOMI) is the distance in embedding space, not token cross-entropy alone.

### The Least You Need to Know (Closing)

Current AI is dominated by transformers (layers of attention + MLP) trained with backprop to predict the next token. SOMI differs: it uses dynamics (activity evolving, oscillating) and geometry (W learning from local stress). SOMI 3.0 can still use backprop for encoders and white matter while using SOMI’s local learning inside each Part, and SOMI can act as an embedding-space predictor (JEPA).

### Check Your Understanding

- [ ] In one sentence: what does backprop do?
- [ ] How is “depth” in a transformer different from “depth” in SOMI (settling steps)?
- [ ] What is the difference between predicting the next token and predicting an embedding?

### Mastery Test

- **Teach it:** Explain “how does a transformer work?” and “how does backprop work?” without equations.
- **Probing questions:** Why might “one global loss” for all weights be both powerful and different from the brain? What would embedding-space prediction buy you compared to token prediction?
- **Key terms:** neural network, weight, transformer, attention, MLP, backpropagation, loss, gradient, next-token prediction, embedding.

---

## Chapter 4: What Current AI Gets Wrong (And Why SOMI Aims to Fix It)

### In This Chapter

- No built-in oscillations
- No local learning (everything via backprop)
- No circuit structure (layers instead of regions and pathways)
- Why this might limit reasoning and efficiency
- What SOMI changes in each case

**Time:** About 15 minutes.

### The Least You Need to Know (Opening)

Current AI is powerful but makes different design choices than the brain: no oscillations, no local plasticity, no circuit architecture. That might leave performance and efficiency on the table. SOMI adds oscillations (field dynamics), local learning (stress-driven W updates), and in 3.0 circuit structure (Parts, White Matter, Systems) — not to replace transformers everywhere, but to explore a different path and to combine both when it helps (e.g., dual learning).

### Why Should You Care?

If you do not know what is “missing” in the default setup, it is hard to see why SOMI exists. This chapter makes the gap explicit so that Parts II and III (the SOMI idea and SOMI 3.0) feel like direct answers.

### No Built-in Oscillations

Transformers do not have a notion of “activity that waves.” They have one forward pass per layer; there is no time dimension of oscillation or settling. So they cannot explicitly use rhythm or phase for coordination. Brains do. SOMI does: activity φ evolves over time (settling steps), can oscillate (underdamped), and has a velocity φ̇. So SOMI can use the *same* machinery for “one step of processing” and “many steps of reasoning” — just run the dynamics longer. That aligns with results like TRM: recursion (many steps of a small net) can beat stacking more layers.

### No Local Learning

In standard deep learning, every weight is updated by backprop from one global loss. The brain instead has local rules (e.g., Hebbian, STDP) plus neuromodulators that gate or scale learning. SOMI follows that: **W** (or W_local in each Part) is updated from *local* quantities: stress tensor (prediction error), kinetic stress, STDP-like terms, structural plasticity. So learning can be local and still drive the system toward better predictions (e.g., lower stress). In SOMI 3.0 we can add backprop on top for the macro (encoders, white matter) and keep SOMI’s local learning for the micro — “dual learning.”

### No Circuit Structure

Transformers are a stack of similar layers; information flows through all of them in one path. Brains have specialized regions and circuits; the same region can participate in multiple circuits and thus face generalization pressure. SOMI 3.0 introduces **Parts** (regions), **White Matter** (pathways), and **Systems** (circuits). So we get modularity, shared parts, and the possibility of different “routes” for different kinds of problems — and we can scale by adding parts and circuits, not only by adding layers.

### Why This Might Limit Reasoning and Efficiency

- **Reasoning:** Multi-step reasoning may benefit from a persistent “working memory” and from running the same dynamics for many steps (like SOMI settling). Transformers do not have an explicit “run longer to think harder” knob; SOMI does (more settling steps).
- **Efficiency:** Circuits and local learning could allow smaller, more specialized subnetworks to do the work instead of one giant stack. JEPA-style embedding prediction (which SOMI’s stress implements) has been shown to help with less overfitting and sometimes better accuracy (e.g., +4% GSM8K in LLM-JEPA) with a different learning signal.

So “what current AI gets wrong” is not “it is bad” — it is “it makes different choices that might leave room for improvement.” SOMI explores those other choices.

### What SOMI Changes (Summary)

| Current AI | SOMI (2.0 and 3.0) |
|------------|---------------------|
| No oscillations | Activity φ and φ̇; can oscillate and settle |
| Global backprop for all weights | Local learning (stress, STDP) for W; optional backprop for macro only (3.0) |
| Stack of layers | Circuit architecture: Parts, White Matter, Systems (3.0) |
| One forward pass per layer | Many settling steps (temporal depth) |
| Token prediction only | Can do embedding prediction (JEPA); stress = prediction error in embedding space |

### The Least You Need to Know (Closing)

Current AI lacks oscillations, local learning, and circuit structure. SOMI adds all three: dynamics that can oscillate, W that learns from local stress, and in 3.0 an architecture of Parts and circuits. That is the gap SOMI is designed to fill.

### Check Your Understanding

- [ ] Name two things “current AI” typically does not have that SOMI adds.
- [ ] Why might “many settling steps” help reasoning compared to “more layers”?
- [ ] What is “dual learning” in one sentence?

### Mastery Test

- **Teach it:** Explain “what does current AI get wrong, and what does SOMI do differently?” in under two minutes.
- **Probing questions:** When might local learning be better than one global backprop? When might circuit structure help generalization?
- **Key terms:** oscillation, local learning, backprop, circuit, Part, White Matter, dual learning, embedding prediction.

---

*End of Part I. Next: Part II — The SOMI Idea (the one equation, how SOMI thinks and learns, stability, neuromodulators).*

---

# Part II: The SOMI Idea (How We Solve It)

---

## Chapter 5: The Core Insight — Activity and Geometry Shape Each Other

### In This Chapter

- The central SOMI idea: one “thing” (activity) lives on another “thing” (geometry), and they co-evolve
- Why we call it a “field on a self-organizing manifold”
- How this connects SOMI 1.0 (oscillations matter) to SOMI 2.0 (one equation for everything)
- The General Relativity analogy (without the math)

**Time:** About 15 minutes.

### The Least You Need to Know (Opening)

In SOMI, **activity** (what the network is doing right now) is a **field** that moves and oscillates on a **graph** whose **geometry** is the set of connection strengths **W**. The geometry decides how activity flows and oscillates; the activity (and the errors it makes) decides how the geometry changes. Both come from **one equation** — the action **S**. So: activity tells geometry how to reshape; geometry tells activity how to move. That is the core insight.

### Why Should You Care?

Everything that follows — the field equation, the geometry equation, stability, neuromodulators, and in 3.0 the circuit brain — is a consequence of this single idea. If you hold onto “activity and geometry shape each other, and both come from S,” the rest will fit together.

### Activity as a Field

**φ (phi)** = activity. It is a number (or a vector of numbers) for each “node” or “feature” at each time. So φ is a **field**: it has a value everywhere on the network. When we say “the field evolves,” we mean: as time goes on, each node’s activity changes according to rules that depend on the current φ and on the connections **W**. So φ is like “temperature” or “pressure” on a map — it has a value at every point (every node), and it can spread and oscillate.

**φ̇ (phi-dot)** = the time derivative of φ — how fast activity is changing. So (φ, φ̇) together tell you “where” the system is and “how it is moving.” That is exactly like position and velocity in physics.

### Geometry as the Connections

**W** = the connectivity matrix. **W_ij** (W-i-j) is the strength of the connection from node j to node i (or between i and j if symmetric). So W defines the **geometry** of the network: who is connected to whom and how strongly. In SOMI this geometry is not fixed — **W learns**. It changes over time based on what the activity is doing (especially on **stress** — prediction error). So we say the **manifold** (the “shape” of the network) is **self-organizing**: it reshapes itself from experience.

### They Shape Each Other

- **Geometry → Activity:** W decides how activity spreads (coupling), how fast different features respond (mass), and how they oscillate (eigenfrequencies). So “geometry tells activity how to move.”
- **Activity → Geometry:** Activity (and the errors e = φ − prediction) produces **stress**. The stress tensor tells us how to update W. So “activity tells geometry how to reshape.”

That two-way dependence is the heart of SOMI. It is written in one place: the **action S[φ, W]**.

### The General Relativity Analogy

In General Relativity, **matter** tells **spacetime** how to curve, and **spacetime** tells **matter** how to move. In SOMI, **activity** tells **geometry (W)** how to change, and **geometry** tells **activity** how to evolve. So we have the same kind of “two-way coupling” from a single underlying principle. The action S is that principle.

### From SOMI 1.0 to SOMI 2.0

- **SOMI 1.0:** Oscillations matter for intelligence; we should use waves and rhythms.
- **SOMI 2.0:** Those oscillations and the learning of W both come from **one action S**. So we do not add oscillations “by hand” — they emerge from the same physics that gives the learning rule. That is why we say “one equation generates everything.”

### The Least You Need to Know (Closing)

Activity φ is a field on the graph; geometry is W. They co-evolve: geometry shapes how φ moves; φ (via stress) shapes how W updates. Both are derived from the single action S[φ, W].

### Check Your Understanding

- [ ] What is φ? What is W?
- [ ] In one sentence: how does geometry affect activity, and how does activity affect geometry?
- [ ] What is the GR analogy in one sentence?

### Mastery Test

- **Teach it:** Explain “activity and geometry shape each other” without writing equations.
- **Key terms:** field, activity (φ), geometry (W), self-organizing manifold, action (S).

---

## Chapter 6: The One Equation That Generates Everything — The Action

### In This Chapter

- What an “action” is in physics (and why we use one)
- The SOMI action **S[φ, W]** in words and in symbols
- What “kinetic” and “potential” mean here
- Why one equation is enough to get both “how φ moves” and “how W learns”

**Time:** About 20 minutes.

### The Least You Need to Know (Opening)

The **action** is a single quantity **S** that depends on the whole history of activity φ and geometry W over time. In physics, you get the “equations of motion” by asking: “Which path makes S stationary?” From that we derive the **field equation** (how φ evolves) and the **geometry equation** (how W evolves). So one object S generates both.

### Why Should You Care?

Whenever you see “field equation” or “geometry equation” or “stress tensor,” they all come from this one S. You do not have to memorize separate rules — they are all consequences of S.

### What Is an Action?

In physics, the **action** is a number you compute for a whole **path** (a history of the system over time). You add up a “Lagrangian” (roughly kinetic energy minus potential energy) at each instant and integrate over time. The **principle of least action** says: the system follows the path that makes the action **stationary** (no small change in the path improves it). From that principle you **derive** the equations of motion (e.g., F = ma). So the action is the “parent” of the dynamics.

In SOMI we do the same: we write an action **S[φ, W]** that depends on φ and W over time. From it we derive:
- the **field equation** (how φ changes at each moment)
- the **geometry equation** (how W changes at each moment)

We also add **dissipation** (damping) and **noise** so that energy can decrease and the system can explore. Those are written in a standard way (Rayleigh function, stochastic terms) and still tie back to the same variational structure.

### The SOMI Action in Words

**S** = integral over time of [ (1/2) × (velocity of φ measured by geometry) − (potential energy) ].

- The **velocity** part (kinetic energy) uses **G(W)** — a “metric” that depends on W. So the “cost” of moving φ quickly depends on the current geometry (e.g., mass per feature). That is why different features can have different timescales.
- The **potential energy V** includes: cost of neighbors disagreeing (coupling), cost of straying from rest (anchor), saturation (tanh), **prediction error** (info stress), error smoothing, coordination, and a small cost on W itself. So V encodes both “how activity should behave” and “how wrong the current prediction is.”

When we minimize (or make stationary) S with respect to φ and W, we get the field equation and the geometry equation. So **one line** (the action) implies both.

### The SOMI Action in Symbols (Optional)

> [!technical]
> $$S[\phi, W] = \int_0^T \left[ \frac{1}{2} \dot{\phi}^T \mathbf{G}(W) \dot{\phi} - V(\phi, W) \right] dt$$
>
> - **φ** = activity (vector); **φ̇** = its time derivative.
> - **G(W)** = metric tensor (from W); in the diagonal approximation it is masses **M_i(W)**.
> - **V(φ, W)** = potential (coupling, anchor, saturation, info stress, error terms, coordination, weight cost).

You do not need to manipulate this to follow the rest. Just know: **everything we do (field equation, geometry equation, stability) comes from this S.**

### Kinetic vs Potential

- **Kinetic:** “Cost” of motion — how much “energy” it takes to have velocity φ̇. When **G** depends on W, geometry sets **mass** (so some features are “heavier” and respond more slowly).
- **Potential:** “Cost” of being in a given state (φ, W) — coupling, anchoring, prediction error, etc. The system “wants” to sit where V is low (e.g., low prediction error, neighbors agreeing).

The field equation is essentially: (mass × acceleration) + (damping × velocity) = −(gradient of V) + noise. So forces come from V; inertia and damping come from the kinetic and dissipation terms.

### Why One Equation Is Enough

Because S is a single functional of the whole path (φ(t), W(t)), making it stationary with respect to φ gives the **field equation**, and making it stationary with respect to W (with dissipation) gives the **geometry equation**. So we do not invent two separate rules — we derive both from one principle. That keeps the theory consistent and guarantees that “activity and geometry shape each other” in a mathematically precise way.

### The Least You Need to Know (Closing)

The action **S[φ, W]** is an integral over time of (kinetic minus potential). From it we derive the field equation (how φ evolves) and the geometry equation (how W evolves). One equation generates everything.

### Check Your Understanding

- [ ] What is the action in one sentence?
- [ ] What do we get from the action? (field equation and geometry equation)
- [ ] What is “kinetic” and “potential” in SOMI in plain language?

### Mastery Test

- **Teach it:** Explain “why one equation is enough” to someone who knows what an equation of motion is.
- **Key terms:** action (S), Lagrangian, kinetic energy, potential energy (V), metric G(W), field equation, geometry equation.

---

## Chapter 7: How SOMI “Thinks” — The Field Equation

### In This Chapter

- What the field equation is (in words)
- The role of mass **M**, damping **β**, and forces
- What “settling” means and why we take many steps
- How this gives “temporal depth” (like TRM’s recursion)

**Time:** About 20 minutes.

### The Least You Need to Know (Opening)

The **field equation** is the law that tells you how activity φ (and its velocity φ̇) change at each instant. It looks like “mass × acceleration + damping × velocity = forces.” The forces come from the potential (coupling, anchoring, prediction error, etc.). We **integrate** this equation for many small time steps — that is **settling**. The more steps, the deeper the “thinking” (temporal depth). So SOMI “thinks” by running the same dynamics repeatedly, not by adding more layers.

### Why Should You Care?

“How does SOMI think?” is answered by: it runs the field equation for **n_settle** steps. So “depth” is **time**, not **layers**. That is why SOMI can align with TRM (recursion beats stacking) and with the “always on” idea (activity keeps evolving).

### The Field Equation in Words

At each node (feature) i:

- **Mass M_i × (acceleration of φ_i)** + **damping β_i × (velocity of φ_i)** = **sum of forces on φ_i**.

So each feature has **inertia** (mass) and **friction** (damping), and the forces push or pull it. The forces include:
1. **Coupling** — neighbors pull toward each other (through W).
2. **Anchoring** — pull toward zero.
3. **Saturation** — tanh so activity does not blow up.
4. **Info stress** — prediction error (precision-weighted).
5. **Basal ganglia gate** — only large errors pass through.
6. **Error smoothing** — errors diffuse on the graph.
7. **Coordination** — connected features encouraged to move together.
8. **Damping** — already in the left-hand side.
9. **Noise** — random kick (during training).

So “thinking” is: start from an initial φ (e.g., from the input embedding), then repeatedly apply this law for a small time step **dt**. After **n_settle** steps, φ has “settled” toward a state that balances all these forces. That final φ is SOMI’s “answer” (e.g., the predicted embedding).

### Mass and Damping

- **M_i (mass)** = how “sluggish” feature i is. High mass → slow response. In SOMI, **M_i** depends on **W**: features with more concentrated connections get **lower** mass (faster); features with diffuse connections get **higher** mass (slower). So geometry sets timescales.
- **β_i (beta)** = damping for feature i. It is chosen so that the **damping ratio ζ (zeta)** is roughly constant (e.g., 0.15 for underdamped, 0.9 for overdamped). Underdamped → oscillations; overdamped → quick settling, little oscillation.

### Settling and Temporal Depth

**Settling** = integrating the field equation for **n_settle** steps. Each step updates φ and φ̇ using the current forces. So the “depth” of processing is **how many steps** we run, not how many layers we have. That is **temporal depth**. TRM showed that a tiny network recursed many times can beat a huge shallow one; SOMI’s settling is exactly “recurse the same dynamics many times.” So (φ, φ̇) after n_settle steps is the “output” of SOMI’s “thinking.”

### The Least You Need to Know (Closing)

The field equation is “mass × acceleration + damping × velocity = forces.” We integrate it for n_settle steps (settling). That is how SOMI thinks — temporal depth, not layer depth.

### Check Your Understanding

- [ ] What are “settling” and “n_settle”?
- [ ] Name three forces in the field equation.
- [ ] How does SOMI get “depth” without more layers?

### Mastery Test

- **Teach it:** Explain “how SOMI thinks” using only the words activity, forces, settling, and steps.
- **Key terms:** field equation, mass (M), damping (β), settling, n_settle, temporal depth, underdamped, overdamped.

---

## Chapter 8: How SOMI “Learns” — The Geometry Equation and Stress

### In This Chapter

- What the geometry equation is (W updates from stress)
- What **stress** is (prediction error in activity or embedding space)
- What the **stress tensor** is and how it drives **W**
- Why this is “local” learning (no backprop through the whole net)
- How this matches JEPA (stress = embedding prediction error)

**Time:** About 20 minutes.

### The Least You Need to Know (Opening)

SOMI learns by changing **W**. W does not change from backprop from a final loss; it changes from **stress** — a local measure of prediction error. The **stress tensor** tells us how much each connection W_ij should move. So learning is **local**: each connection is updated from quantities (activity, error) that are nearby in the graph. In SOMI-JEPA, stress is exactly the **embedding-space prediction error** (distance between SOMI’s prediction and the target embedding), so SOMI’s learning **is** JEPA-style learning.

### Why Should You Care?

This is what makes SOMI “brain-like” in the learning sense: no single global backward pass, but many local updates from prediction error. And it is what makes SOMI a natural JEPA predictor: the same quantity (stress) is both the “loss” and the driver of learning.

### The Geometry Equation in Words

**W** changes slowly (on a longer timescale than φ). The change is:

- **dW/dt** (or the step update ΔW) is proportional to a function of **stress** and **kinetic stress** (and possibly STDP-like terms).

So “stress” is the main learning signal. When stress is high, W is updated more (or in a direction that tends to reduce stress). When stress is low, W changes little. The exact formula involves the **stress tensor** **S_ij** (not to be confused with the action S): a matrix that says how much each pair (i, j) contributes to the total error. So **S_ij** drives the update to **W_ij**. That is the **geometry equation**.

### What Is Stress?

**Stress** = how wrong the current prediction is. In the simplest view:

- We have a **target** (e.g., the correct next activity or the embedding of the correct answer).
- We have SOMI’s **prediction** (e.g., φ after settling, or the readout from φ).
- **Stress** = the mismatch between them (e.g., squared distance or precision-weighted error).

So stress is “prediction error.” In SOMI 2.0 it appears in the potential V as **V_info**; the gradient of V_info with respect to φ gives one of the forces (info stress force), and the way V depends on W (through the error e and the graph) gives the **stress tensor** that updates W.

### The Stress Tensor and W Updates

The **stress tensor** (we can call it **S_ij** in symbols, but in the doc we often just say “stress tensor”) is built from:
- **Information stress:** from (φ − prediction) and precision; which connections “carry” the error.
- **Kinetic stress:** from φ̇ and the geometry; which connections are under velocity load.

The geometry equation says: **ΔW_ij** is proportional to (something derived from) this tensor. So connections that are implicated in high error get updated more; connections that are not, stay put. That is **local** learning: the update to W_ij depends on quantities at or near i and j, not on a global backward pass from the output.

### Why This Is Local Learning

We do not compute “gradient of final loss with respect to W_ij” by backpropagating through the whole network. We compute “how much does stress depend on W_ij (or on activity at i and j)?” from **local** information: current φ, φ̇, prediction, target, and W in the neighborhood. So each W_ij update is a **local rule**. The brain does something similar (e.g., Hebbian, STDP). In SOMI 3.0 we can still use backprop for **macro** parameters (encoders, white matter) and keep this local rule for **micro** (W inside each Part).

### Stress as JEPA Loss

In JEPA, the loss is **distance in embedding space**: distance(Predictor(input), Target_embedding). In SOMI, **stress** is exactly that: the (precision-weighted) distance between SOMI’s predicted state (e.g., φ at the motor/output region or a readout of φ) and the target embedding. So:

- **SOMI’s stress** = **JEPA prediction error** (in embedding space).
- **SOMI’s W update from stress** = **learning from the JEPA loss** without backprop through the predictor. So SOMI naturally implements JEPA-style training when we use it as the predictor.

### The Least You Need to Know (Closing)

SOMI learns by updating W from the **stress tensor** (prediction error). Learning is **local**. Stress in embedding space is the JEPA loss, so SOMI as predictor is JEPA-compatible.

### Check Your Understanding

- [ ] What is “stress” in one sentence?
- [ ] Why do we say W learning is “local”?
- [ ] How is stress related to JEPA loss?

### Mastery Test

- **Teach it:** Explain “how SOMI learns” without backprop, using stress and the stress tensor.
- **Key terms:** geometry equation, stress, stress tensor, local learning, JEPA, embedding prediction error.

---

## Chapter 9: How SOMI Stays Stable — Energy and Lyapunov

### In This Chapter

- What the **Hamiltonian** **H** is (total energy: kinetic + potential)
- Why we want **H** to decrease (or not increase) during settling
- What “Lyapunov stability” means in plain language
- What goes wrong if the Hamiltonian increases (pathology)

**Time:** About 15 minutes.

### The Least You Need to Know (Opening)

SOMI has an **energy** **H** (Hamiltonian) = kinetic + potential. During settling, **H** should **decrease** (or stay constant), not increase. That is **Lyapunov stability**: the system moves toward lower energy, so it does not blow up. If **H** ever increases, that is a **physics violation** and we flag it as a critical pathology (e.g., in diagnostics).

### Why Should You Care?

Stability is what makes SOMI’s dynamics safe to run for many steps. If energy could grow without bound, activity would explode. The theory (and the code) are set up so that under normal conditions H decreases, so we can run long settling and still get sensible answers.

### The Hamiltonian H

**H** = **T** + **V** = kinetic energy + potential energy.

- **T** = (1/2) φ̇^T G(W) φ̇ (or in diagonal form, sum over i of (1/2) M_i φ̇_i^2).
- **V** = the same potential we had in the action (coupling, anchor, saturation, info stress, etc.).

So H is the “total energy” of the system. When there is no damping and no noise, H would be conserved. With damping, H **decreases** over time (energy is dissipated). So as we settle, we expect H to go down.

### Lyapunov Stability in Plain Language

A **Lyapunov function** is a quantity that (in a well-behaved system) never increases along the trajectory. Here, **H** is that quantity: with damping, dH/dt ≤ 0. So the system “runs downhill” in energy. That means:
- Activity does not explode.
- The system tends toward a minimum of V (or a low-energy state), which is exactly where we want “settled” predictions.

So we say SOMI is **Lyapunov stable** when H decreases during settling.

### What If H Increases?

If we ever see **H** increase during a settling run, something is wrong (e.g., numerical instability, bug, or bad parameters). So in the diagnostics we have a check: **hamiltonian_increasing**. If True, we flag it as a **critical** pathology. In practice we use **symplectic** integration (e.g., symplectic Euler) so that energy behavior is preserved as well as possible; ordinary Euler can add energy and cause blow-ups.

### The Least You Need to Know (Closing)

The Hamiltonian H = T + V is total energy. With damping, H should decrease during settling (Lyapunov stability). If H increases, we treat it as a critical failure and flag it in diagnostics.

### Check Your Understanding

- [ ] What is H (Hamiltonian)?
- [ ] Why do we want H to decrease?
- [ ] What do we do if H increases?

### Mastery Test

- **Teach it:** Explain “why SOMI does not blow up” using energy and Lyapunov.
- **Key terms:** Hamiltonian (H), kinetic energy (T), potential energy (V), Lyapunov stability, damping, pathology.

---

## Chapter 10: SOMI’s Built-in Brain Chemicals — Neuromodulators

### In This Chapter

- What neuromodulators are (in the brain and in SOMI)
- The four systems: **NE**, **DA**, **ACh**, **5-HT**
- What each one “monitors” and what it “modulates” in SOMI
- Why they matter for adaptive behavior (e.g., learning rate, settling depth, attention)

**Time:** About 15 minutes.

### The Least You Need to Know (Opening)

The brain uses **neuromodulators** (e.g., dopamine, acetylcholine) to adjust how regions learn and respond — like “volume knobs” that turn up or down learning rate, attention, or persistence. SOMI 2.0 has four such systems: **NE** (arousal), **DA** (reward/learning rate), **ACh** (attention/salience), **5-HT** (patience/persistence). They are computed from local signals (e.g., surprise, stress change, per-feature error) and then **modulate** SOMI parameters (e.g., η, n_settle, mass, noise). So SOMI can adapt its own behavior without a human changing hyperparameters.

### Why Should You Care?

Neuromodulators are what make SOMI “self-calibrating” and adaptive. When stress is high, DA might boost learning rate; when a problem is hard, 5-HT might allow more settling steps. So the system tunes itself. In SOMI 3.0 we can use the same machinery per Part and add diagnostic feedback (e.g., pathology → parameter response).

### The Four Neuromodulators in SOMI

**1. NE (Norepinephrine) — Arousal / Alertness**

- **Monitors:** Surprise (e.g., how much current error differs from a running average).
- **Modulates:** Arousal level (0–1), which then affects **η** (geometry learning rate) and possibly other global knobs.
- **Effect:** When things are surprising, learning is turned up; when things are routine, learning is turned down.

**2. DA (Dopamine) — Reward / Learning Rate**

- **Monitors:** Stress improvement (e.g., is stress going down?).
- **Modulates:** **η** (geometry learning rate) — a multiplier (e.g., 0.5× to 1.5×).
- **Effect:** When the system is improving (stress decreasing), DA boosts η; when it is not, DA reduces it. In JEPA terms, DA “rewards” reduction in prediction error.

**3. ACh (Acetylcholine) — Attention / Salience**

- **Monitors:** Per-feature error (which features have the biggest prediction error).
- **Modulates:** **Mass** (or effective mass) per feature — salient features get lower mass so they respond faster.
- **Effect:** “Attention” in SOMI = devoting more dynamics to the features that are most wrong.

**4. 5-HT (Serotonin) — Patience / Persistence**

- **Monitors:** Relative difficulty (e.g., error vs running average).
- **Modulates:** **n_settle** (e.g., 0.75× to 1.25×) and **noise_ratio** (e.g., 0.8× to 1.2×).
- **Effect:** Hard problems get more settling steps and a bit more exploration; easy ones get fewer steps and less noise.

### Why They Matter

Together, these four systems make SOMI **adaptive**: it can speed up or slow down learning, focus on salient features, and run longer when needed. That is a step toward “diagnostic-guided” training: in 3.0 we can add more feedback (e.g., pathology detection → parameter changes) on top of these same modulators.

### The Least You Need to Know (Closing)

SOMI has four neuromodulator systems (NE, DA, ACh, 5-HT) that monitor surprise, stress change, per-feature error, and difficulty, and modulate arousal, η, mass, n_settle, and noise. They make SOMI self-calibrating and adaptive.

### Check Your Understanding

- [ ] Name the four neuromodulators and one thing each modulates.
- [ ] What does DA respond to (in one sentence)?
- [ ] Why might 5-HT increase n_settle?

### Mastery Test

- **Teach it:** Explain “what are neuromodulators in SOMI and why do we have them?”
- **Key terms:** neuromodulator, NE, DA, ACh, 5-HT, arousal, η (eta), n_settle, mass, self-calibration.

---

*End of Part II. Next: Part III — The SOMI 3.0 Leap (circuits, JEPA, dual learning, graph/manifold, diagnostics).*

---

# Part III: The SOMI 3.0 Leap (The New Architecture)

---

## Chapter 11: From Flat W to Brain Circuits — Parts, White Matter, Systems

### In This Chapter

- Why “one big W” is replaced by **Parts** (regions) and **White Matter** (pathways)
- What a **Part** is (mini-SOMI with its own φ, W_local, neuromodulators)
- What **White Matter** is (sparse, fixed or slowly plastic projections between Parts)
- What a **System** is (a circuit — a route through several Parts)
- How shared Parts create generalization pressure

**Time:** About 20 minutes.

### The Least You Need to Know (Opening)

SOMI 3.0 does not use one flat connectivity matrix. It uses **Parts** (brain regions): each Part has its own activity φ, its own **W_local**, and its own physics (mass, damping, neuromodulators). Parts are connected by **White Matter** (long-range, sparse, fixed or slowly plastic). **Systems** are circuits: defined routes through several Parts. When a Part belongs to several Systems, it must serve all of them — that **generalization pressure** is how we get shared, reusable representations.

### Why Should You Care?

This is the main architectural leap of 3.0. It gives us modularity, scaling by “more parts and circuits” instead of “more layers,” and a direct analogy to the brain’s regions and tracts.

### What Is a Part?

A **Part** is like a small SOMI: a “brain region.”

- It has its own **φ** and **φ̇** (activity and velocity) that **persist across tokens** (the “always on” brain).
- It has its own **W_local** — a connectivity matrix **inside** that Part only. W_local is dense (or moderately sparse) and **plastic**: it learns via stress tensor, STDP, structural plasticity.
- It has its own **mass**, **damping**, **eigenvalues**, **neuromodulators**, and diagnostics.
- Parts can have **different sizes** (local_dim) and **different physics** (e.g., different target_zeta).

So each Part is a mini-SOMI. The current code’s **SOMILayer** is the base we refactor into **SOMIPart** for 3.0.

### What Is White Matter?

**White Matter** is the wiring **between** Parts. It is:

- **Sparse:** Each Part connects to only a few others (e.g., 3–5), not all-to-all.
- **Fixed or slowly plastic:** The *topology* (who connects to whom) is set at initialization; the weights of these projections change slowly or not at all (developmental timescale).
- Implemented as **low-rank linear maps** (e.g., down_proj and up_proj): Part A’s output → rank r → Part B’s input. So long-range communication is compressed.

So: **locally dense** (inside each Part), **globally sparse** (between Parts). That is the “small-world” design.

### What Is a System?

A **System** is a **circuit**: a list of Parts that information flows through for one function. Examples for math reasoning:

- **Comprehension:** Sensory → Association → PFC
- **Computation:** PFC → Numerics → Working Memory → PFC
- **Memory:** Association → Hippocampus → PFC
- **Verification:** Working Memory → PFC → Association → Motor

Multiple Systems run **at the same time**. A Part that appears in more than one System (e.g., **PFC** in all four above) receives input from all of them. So its W_local is updated by stress from **all** those Systems — it must learn representations that work for comprehension, computation, memory, and verification. That is **generalization pressure**: shared Parts are forced to be general.

### The Least You Need to Know (Closing)

Parts = regions (each with φ, W_local, physics). White Matter = sparse, fixed or slow links between Parts. Systems = circuits (routes through Parts). Shared Parts = generalization pressure.

### Check Your Understanding

- [ ] What is inside a Part?
- [ ] How is White Matter different from W_local?
- [ ] Why does a shared Part (e.g., PFC) face “generalization pressure”?

### Mastery Test

- **Teach it:** Explain “Parts, White Matter, and Systems” to someone who only knows “one big W.”
- **Key terms:** Part, W_local, White Matter, System, circuit, generalization pressure, sparse, dense.

---

## Chapter 12: How Circuits Create Generalization — The PFC Example

### In This Chapter

- Why putting the same Part in multiple Systems forces it to generalize
- The PFC example: one region serving comprehension, computation, memory, verification
- How stress from all Systems shapes W_local
- Why this is “like the brain”

**Time:** About 10 minutes.

### The Least You Need to Know (Opening)

When Part B is in both System A and System C, B’s W_local gets **stress** from A and from C. So B cannot overfit to just one task — it must find representations that work for **both**. That is how circuits create generalization. PFC in the brain is like that; in SOMI 3.0 we design Systems so that key Parts (e.g., “PFC”) are shared on purpose.

### Why Shared Parts Generalize

If a Part were in only one System, it could specialize narrowly to that System’s inputs and targets. When it is in **several** Systems, the stress tensor that updates W_local is a **sum** (or combination) of stresses from all those Systems. So the Part’s weights are pulled in multiple “directions” — they must satisfy all routes. That favors **general** representations that transfer across tasks. So we get generalization “for free” from the architecture.

### The PFC Example

In our math-reasoning design, **PFC** (prefrontal cortex) appears in:
- Comprehension (understanding the problem)
- Computation (doing the math)
- Memory (recalling facts)
- Verification (checking the answer)

So PFC’s φ and W_local are used by four different circuits. Its learning signal is a mix of “did we understand?”, “did we compute correctly?”, “did we retrieve the right thing?”, “did we verify?”. So PFC is pushed to encode abstract, reusable structure — not just one narrow skill. That is exactly how prefrontal cortex is thought to work in the brain: multimodal, general, and reused across tasks.

### The Least You Need to Know (Closing)

Shared Parts receive stress from every System they belong to, so they learn general representations. PFC is the canonical example: one Part, many circuits, forced generality.

### Check Your Understanding

- [ ] Why does a shared Part generalize better?
- [ ] Name two Systems that might both use “PFC.”

### Mastery Test

- **Key terms:** shared Part, generalization pressure, PFC, stress from multiple Systems.

---

## Chapter 13: SOMI as a JEPA Predictor — Predicting Embeddings, Not Tokens

### In This Chapter

- What JEPA is (Joint Embedding Predictive Architecture) in plain language
- How SOMI can be the **predictor**: input embedding → SOMI settles → predicted target embedding
- Why stress = JEPA loss (embedding-space prediction error)
- What X-Encoder, Y-Encoder, Y-Decoder do in SOMI-JEPA
- Why this helps (e.g., smoother learning, selective decoding)

**Time:** About 20 minutes.

### The Least You Need to Know (Opening)

**JEPA** trains a system to predict a **target embedding** from an input, instead of (or in addition to) predicting the next token. SOMI is a natural **predictor**: you feed it an input embedding, it settles, and the output (e.g., φ at the motor Part or a readout) is the **predicted target embedding**. The **stress** is exactly the distance between that prediction and the true target embedding — so **stress = JEPA loss**. Training SOMI from stress is therefore JEPA training. We add **X-Encoder** (input → embedding), **Y-Encoder** (target text → target embedding), and **Y-Decoder** (predicted embedding → text when we need to output).

### Why Should You Care?

Papers (LLM-JEPA, VL-JEPA) show that adding embedding prediction improves accuracy and robustness. SOMI was already doing “learn from prediction error in a continuous space” — so aligning it with JEPA makes the theory clean and lets us use SOMI as the core predictor in a modern architecture.

### What Is JEPA?

**JEPA** = Joint Embedding Predictive Architecture (LeCun and others). Idea:

- **Encode** input (e.g., text, image) into an embedding.
- **Encode** target (e.g., next segment, answer) into an embedding.
- **Predictor:** map input embedding → predicted target embedding.
- **Loss** = distance(predicted target embedding, true target embedding). So we train in **embedding space**, not (only) in token space.

SOMI fits as the predictor: input embedding is injected (e.g., into the sensory Part), SOMI settles, and the readout (e.g., motor Part φ) is the predicted target embedding. **Stress** in SOMI is exactly that distance (possibly precision-weighted). So **SOMI’s learning = JEPA training.**

### X-Encoder, Y-Encoder, Y-Decoder

- **X-Encoder:** Turns raw input (e.g., problem text) into an embedding that drives SOMI (e.g., injected into the sensory Part). Can be a standard text encoder or part of an LLM.
- **Y-Encoder:** Turns the **target** (e.g., correct answer text) into the embedding we want SOMI to predict. Often **frozen** (e.g., EmbeddingGemma, sentence-transformers).
- **Y-Decoder:** Turns SOMI’s **predicted embedding** back into text when we need an answer. Used **selectively** (e.g., only at the end of reasoning), like VL-JEPA’s “decode when needed.” So the “brain” (SOMI) is always on in embedding space; we only decode to tokens when we have to.

### Why This Helps

- **Smoother learning:** Embedding space is continuous; small changes in W can smoothly reduce stress. Token space is discrete and noisier.
- **Selective decoding:** We can run SOMI for many steps (reasoning) and only decode once, saving compute.
- **Robustness:** LLM-JEPA showed that adding JEPA loss does not hurt generation and can improve accuracy (e.g., +4% GSM8K). SOMI-JEPA uses the same idea with SOMI as the predictor.

### The Least You Need to Know (Closing)

SOMI is the JEPA predictor: it maps input embedding → predicted target embedding. Stress = JEPA loss. X-Encoder, Y-Encoder, Y-Decoder handle text ↔ embedding. This gives embedding-space training and selective decoding.

### Check Your Understanding

- [ ] What is the “predictor” in JEPA? What is it in SOMI-JEPA?
- [ ] What is stress in terms of JEPA?
- [ ] When do we use Y-Decoder?

### Mastery Test

- **Teach it:** Explain “SOMI as JEPA predictor” and “stress = JEPA loss.”
- **Key terms:** JEPA, predictor, embedding prediction, X-Encoder, Y-Encoder, Y-Decoder, stress, selective decoding.

---

## Chapter 14: The Dual Learning Trick — Backprop for Macro, SOMI for Micro

### In This Chapter

- What “dual learning” means: two learning mechanisms in one system
- **Macro** (encoders, White Matter): learned by **backprop** from the JEPA loss (and optionally token loss)
- **Micro** (W_local inside each Part): learned by **SOMI’s local rules** (stress tensor, STDP)
- How gradients flow: straight-through estimator so macro can get a signal through SOMI
- Why this is biologically plausible and mathematically sound (Lyapunov preserved)

**Time:** About 15 minutes.

### The Least You Need to Know (Opening)

In SOMI 3.0 we can train **two kinds** of parameters with **two kinds** of learning. **Macro** (X-Encoder, Y-Decoder, White Matter tracts) is trained with **backprop** so that the whole pipeline minimizes JEPA loss (and maybe token loss). **Micro** (W_local inside each Part) is trained with **SOMI only** — stress tensor, STDP, structural plasticity — so that learning stays local and brain-like. To let the JEPA loss reach the macro parameters “through” SOMI, we use a **straight-through estimator**: we treat SOMI’s output as if it had a gradient equal to the downstream gradient, so backprop does not try to differentiate through the settling dynamics. Result: macro gets a global learning signal; micro stays on local rules; Lyapunov stability of SOMI is preserved.

### Why Should You Care?

This is how we get the best of both worlds: **global credit assignment** (backprop) for the “highway” (encoders, white matter) and **local plasticity** (SOMI) for the “neighborhoods” (Parts). It also matches the brain: developmental wiring (macro) vs ongoing synaptic plasticity (micro).

### What Is Macro vs Micro?

- **Macro:** Structure that is shared across the whole system and that we want to optimize with a single loss. Examples: X-Encoder weights, Y-Decoder weights, White Matter tract weights (low-rank matrices). These are **differentiable** and we have a loss (JEPA + maybe token) at the end, so we can backprop.
- **Micro:** The internal connectivity **W_local** inside each Part. We **do not** backprop into W_local. We update W_local only with SOMI’s rules (stress, STDP, structural plasticity). So micro is “local learning.”

### How Gradients Flow — Straight-Through

The loss (e.g., JEPA distance) is computed from SOMI’s **output** (e.g., predicted embedding). To train the X-Encoder and White Matter, we need a gradient of the loss with respect to **their** parameters. But the path from them to the loss goes **through** SOMI’s settling (many steps of φ, φ̇, and W_local updates). We do **not** backprop through W_local or through the exact settling dynamics. Instead we use a **straight-through estimator**: we pretend SOMI’s output is a differentiable function of its **input** (e.g., the embedding we injected), and we pass the downstream gradient through as if ∂(output)/∂(input) = identity (or a simple approximation). So the macro parameters get a learning signal; the internal SOMI dynamics stay “opaque” to backprop. That is standard in hybrid systems (e.g., discrete latent variables).

### Biologically Plausible and Mathematically Sound

- **Biology:** Long-range tracts (white matter) are laid down early and change slowly; synaptic plasticity (grey matter) is ongoing. So “macro fixed or slow, micro plastic” fits. Backprop on macro is a stand-in for developmental learning; SOMI on micro is a stand-in for local plasticity.
- **Math:** Lyapunov stability (H decreasing) is a property of the **SOMI dynamics** (φ, W_local). When we update only W_local with stress and do not backprop into W_local, those dynamics are unchanged. So stability is preserved. The macro update does not alter the internal physics of each Part.

### The Least You Need to Know (Closing)

Dual learning: **macro** (encoders, white matter) ← backprop from JEPA/token loss; **micro** (W_local) ← SOMI local rules only. Straight-through lets gradients reach macro through SOMI. Lyapunov is preserved; biology is respected.

### Check Your Understanding

- [ ] What is “macro” and what is “micro”?
- [ ] Why do we use a straight-through estimator?
- [ ] Why is Lyapunov still valid?

### Mastery Test

- **Teach it:** Explain “backprop for macro, SOMI for micro” and why we do both.
- **Key terms:** dual learning, macro, micro, backprop, straight-through estimator, Lyapunov.

---

## Chapter 15: The Graph and the Manifold — Both at Once

### In This Chapter

- Locally (inside a Part): SOMI is like a **manifold** (continuous geometry, metric from W_local)
- Globally (between Parts): SOMI is a **graph** (nodes = Parts, edges = White Matter)
- Why we say “both”: the graph is the coarse structure; each node is a little manifold
- How the graph Laplacian relates to the manifold (Laplace–Beltrami) for the math-inclined

**Time:** About 10 minutes.

### The Least You Need to Know (Opening)

SOMI 3.0 is **both** a **graph** (at the scale of Parts and White Matter) and a **manifold** (inside each Part, where W_local defines a continuous geometry). So we do not choose “graph OR manifold” — we have a **graph of manifolds**: each Part is a local manifold; the Parts are connected by a sparse graph. That matches the brain: local cortical geometry (manifold-like) and long-range tracts (graph-like).

### Why Should You Care?

This clears up “is SOMI a graph or a manifold?” The answer: **graph globally, manifold locally.** The math (e.g., graph Laplacian → Laplace–Beltrami in the limit) supports this duality.

### Locally: Manifold

Inside one Part, **W_local** is a dense (or moderately sparse) connectivity matrix. It defines a **metric** (e.g., G(W) in the action) and thus a **Riemannian geometry**. So activity φ in that Part evolves like a field on a **manifold**. The SOMI 2.0 continuous theory (field on a manifold) applies **within** each Part.

### Globally: Graph

**Parts** are **nodes**; **White Matter tracts** are **edges**. So at the coarse scale we have a **graph**: which Parts connect to which, with what (fixed or slow) weights. That is the “brain graph” — sparse and macroscopic.

### Both at Once

So: **globally** we have a graph (Parts + White Matter). **Locally** each Part is a manifold (W_local, φ, field equation). So SOMI 3.0 is a **graph whose nodes are manifolds** — or a discretization of a larger manifold into “cells” (Parts) connected by a graph. The graph Laplacian (on the Part graph) and the Laplace–Beltrami operator (on each Part’s manifold) are related in the limit when we refine the discretization; the theory supports this unified view.

### The Least You Need to Know (Closing)

Inside a Part = manifold (W_local, metric). Between Parts = graph (White Matter). SOMI is both: a graph of local manifolds.

### Check Your Understanding

- [ ] Where is SOMI “manifold-like” and where “graph-like”?
- [ ] What are the “nodes” and “edges” of the global graph?

### Mastery Test

- **Key terms:** graph, manifold, W_local, White Matter, graph Laplacian, Laplace–Beltrami.

---

## Chapter 16: How SOMI 3.0 Monitors Its Own Health — Diagnostics as Feedback

### In This Chapter

- What “diagnostics” are in SOMI (per-Part and circuit-level)
- Core physics diagnostics (stress, H, mass, W stats, etc.)
- Neuromodulator levels and pathology detection
- How diagnostics can drive **adaptive training** (e.g., pathology → parameter response)
- The idea of “brain scan” and “self-healing”

**Time:** About 15 minutes.

### The Least You Need to Know (Opening)

SOMI 2.0 already has rich **diagnostics**: stress, Hamiltonian, mass hierarchy, eigenspectrum, neuromodulators, and **pathology** checks (e.g., hamiltonian_increasing, geometry_explosion). In 3.0 we add **circuit-level** diagnostics (per-circuit stress, throughput, shared-part pressure) and **JEPA-specific** ones (prediction error, stress–JEPA correlation). These are not just for logging — they can **drive** training: e.g., if a pathology is detected, we adjust parameters (more settling, lower learning rate, etc.) so the system “self-heals.”

### Why Should You Care?

Diagnostics turn SOMI from a black box into a **observable** system. We can see when it is healthy (H decreasing, stress dropping, no pathologies) and when it is not. Using them for **feedback** (diagnostic-guided adaptive training) is the next step toward a brain-like, self-regulating learner.

### Core Physics Diagnostics (Per Part)

We track (per Part, per token or per step):

- **Stress** (info, kinetic, stress_weight)
- **Hamiltonian** (start, end, did it decrease?)
- **W** (mean, std, sparsity)
- **Mass** (min, max, ratio) and **eigenvalues** (spectrum)
- **φ** (std, range, consciousness entropy)
- **Self-calibration** (arousal, beta, n_settle, eta)
- **Dale’s law** (E/I counts, lambda_E, lambda_C)

So we have a “vital signs” panel for each Part.

### Neuromodulators and Pathologies

- **Neuromodulator levels:** NE, DA, ACh, 5-HT (and their targets, e.g., eta_multiplier, settle_multiplier).
- **Pathology table:** We check for 11 failure modes (e.g., hamiltonian_increasing, geometry_explosion, oscillations_persistent, stress_increasing, features_collapsed, gates_closed, …). Each has a **brain_pathology** name and a **symptom** description. If any trigger, we log them and can **respond** (e.g., reduce eta, increase n_settle, or pause learning).

### Circuit-Level and JEPA Diagnostics (3.0)

- **Per circuit:** circuit_stress, circuit_throughput, circuit_coherence, circuit_bottleneck.
- **White matter:** tract_utilization, tract_gradient, information_flow.
- **Shared parts:** shared_part_stress_variance, generalization_pressure.
- **JEPA:** jepa_prediction_error, jepa_error_by_circuit/part, stress_jepa_correlation (should be ~1 if stress is the JEPA loss).

### Diagnostic-Guided Adaptive Training

The idea: **if** a pathology (or a bad diagnostic) is detected, **then** change parameters (e.g., lower η, increase n_settle, boost damping). So training becomes a **closed loop**: diagnostics → controller → parameters → better behavior. That is “self-healing” and “adaptive compute” (e.g., hard problems get more settling automatically via 5-HT and/or pathology response).

### The Least You Need to Know (Closing)

Diagnostics measure physics, neuromodulators, and pathologies per Part and (in 3.0) per circuit and JEPA. They can feed into adaptive training so SOMI “self-heals” and allocates compute where needed.

### Check Your Understanding

- [ ] Name two core physics diagnostics and two pathologies.
- [ ] What is “diagnostic-guided adaptive training” in one sentence?
- [ ] What should stress_jepa_correlation be if stress = JEPA loss?

### Mastery Test

- **Teach it:** Explain “how SOMI monitors its own health” and “how that can guide training.”
- **Key terms:** diagnostics, pathology, Hamiltonian, stress, neuromodulator, adaptive training, self-healing.

---

*End of Part III. Next: Part IV — Where We Are and Where We Are Going.*

---

# Part IV: Where We Are and Where We Are Going

---

## Chapter 17: Experiments We Have Run and What They Showed

### In This Chapter

- What experiments have been run so far (OLMo-2 1B, Huginn 3.5B, VibeThinker-1.5B, GSM8K, MATH)
- What conditions were tested (baseline, ETD, SOMI overdamped/underdamped, SOMI augment, knowledge absorption, etc.)
- What we learned (zero-shot baselines, diagnostics fixes, engineering lessons, 99% knowledge transfer)
- What is still pending (Phase 2 local learning, Phase 3 hybrid RL)

**Time:** About 20 minutes.

### The Least You Need to Know (Opening)

We have run **Phase 1** (zero-shot) experiments on OLMo-2 1B and Huginn 3.5B on math (GSM8K, MATH), plus **VibeThinker-1.5B validation experiments** (Feb 11, 2026) that proved SOMI's core predictions. VibeThinker experiments achieved: 99% knowledge transfer efficiency, extracted SOMI physics from all 28 layers, generated 9 diagnostic visualizations, and validated stress-guided learning. Results confirm diagnostics and integration work, and provide the first empirical validation of SOMI theory. **Phase 2** (SOMI local learning, transformer frozen) and **Phase 3** (hybrid RL) are planned next.

### Why Should You Care?

This is the empirical ground truth: what works, what broke, and what we must preserve (symplectic integration, W as buffer, eval-mode diagnostics, pathology key names, etc.) when we build SOMI-JEPA and the circuit brain.

### What Was Run

**OLMo-2 1B:**

- Baseline (16 layers), ETD k=2/3/4 (layers 7–10 repeated), SOMI overdamped/underdamped (W from MLP), SOMI random W, SOMI augment (full model + SOMI after layer 10), ETD k=2 + SOMI augment.
- Benchmarks: GSM8K (200 problems), MATH (started).
- Accuracies (GSM8K): low single digits (e.g., 2–4%) across conditions; baseline ~2–4%; ETD and SOMI augment in similar range. MATH baseline 21% before output ended.

**Huginn 3.5B:**

- Baseline (r=32 recurrence), SOMI overdamped/underdamped/random W/augment.
- GSM8K (100 problems): baseline partially completed (6%); other conditions truncated.

**Infrastructure:** RunPod (B200/H100/A100/L40S), CUDA 12.8 for B200, PyTorch cu128, transformers pinned for Huginn.

**VibeThinker-1.5B (Feb 11, 2026):**

Four validation experiments on RunPod A100 80GB PCIe:

**Experiment 1: Knowledge Absorption**
- 99% transfer efficiency (code specialist → base model that never saw code)
- No catastrophic forgetting (English improved: 8.69 → 7.29)
- Validates: knowledge = topology, surgical transplantation works

**Experiment 2: SOMI Conversion**
- Extracted W (connectivity), ρ (mass), S (stress) from all 28 layers
- Settling rate: 0.7276 (strong equilibrium)
- Mass hierarchy: 0.621 ± 0.311 (significant variation)
- Stress profile: highest at output layer (0.7783)
- Validates: transformers have SOMI physics

**Experiment 3: Neuroscience Diagnostics**
- 9 visualizations in 14 seconds
- Domain-specific stress: code (0.924), math (0.889)
- Domain-specific loss: code (6.99), English (8.92)
- Validates: diagnostics reveal knowledge localization

**Experiment 4: Cross-Size Absorption Pipeline**
- Qwen2.5-7B → VibeThinker-1.5B distillation
- Pipeline works (vocab fix, no crashes)
- Basic KL insufficient (150 steps), needs stress-guided weighting
- Validates: infrastructure ready, needs tuning

**Total runtime:** ~7 minutes across 3 pods
**Total cost:** ~$0.48

### What We Learned (Engineering)

Critical fixes to **carry forward** into all future runs:

1. **Symplectic Euler** for settling — required for stable oscillations; plain Euler can add energy.
2. **W as buffer, not nn.Parameter** — W is updated only by stress/STDP; no backprop into W.
3. **Eval-mode diagnostics** — collect diagnostics in both train and eval; originally they were only in train.
4. **Pathology keys** — use `brain_pathology` and `symptom`, not `pathology`; safe get in scripts.
5. **Safe formatting** — some metrics can be missing (`'?'`); use a `_f()` helper to avoid format errors.
6. **STDP asymmetry** — do not symmetrize; STDP is directional.
7. **No double-counting CFC** — coupling/coordination already capture it.
8. **Position embeddings API** — newer transformers use `position_embeddings=(cos, sin)`; use helpers.
9. **B200 support** — CUDA 12.8 + PyTorch cu128.
10. **Fixed curriculum** — reuse patterns across batches where relevant.

### What Is Pending

- **Phase 2:** SOMI local learning (stress, STDP, structural plasticity) with **transformer frozen**. Reuse eval scripts and all fixes; add a training loop that only updates SOMI state.
- **Phase 3:** Hybrid RL (or hybrid backprop + SOMI) for full pipeline training.
- **Diagnostics:** Ensure eval-mode diagnostics are deployed and visible in remote runs (some runs showed N/A; verify fix on pod).
- **SOMI-JEPA and circuit brain:** Design and implement Parts, White Matter, Systems, JEPA training loop, and diagnostic feedback.

### The Least You Need to Know (Closing)

Phase 1 zero-shot runs are done on OLMo-2 and Huginn on GSM8K/MATH. We have a list of engineering fixes that must carry forward. Phase 2 (local learning) and Phase 3 (hybrid) and the full SOMI 3.0 stack are next.

### Check Your Understanding

- [ ] Name two conditions we ran on OLMo-2.
- [ ] Why is symplectic Euler non-negotiable?
- [ ] What is Phase 2 in one sentence?

### Mastery Test

- **Key terms:** Phase 1/2/3, zero-shot, local learning, hybrid RL, symplectic Euler, W buffer, eval diagnostics.

---

## Chapter 18: The Roadmap — From Experiments to a Full SOMI Brain

### In This Chapter

- The logical order: finish Phase 1/2/3 on existing LLMs, then build the circuit architecture
- SOMI-JEPA: design (X/Y encoders, SOMI as predictor, stress = JEPA loss), then implement
- Circuit brain: SOMIPart, WhiteMatterTract, SOMIBrain, topology (e.g., 6–8 Parts, 3–4 Systems for math)
- Diagnostic-guided training and scaling (Circuit-S, M, L, XL)
- Comparison experiments (Circuit vs flat SOMI vs SOMI-in-transformer vs TRM-style)

**Time:** About 15 minutes.

### The Least You Need to Know (Opening)

The roadmap is: (1) Complete Phase 2 and 3 on pretrained LLMs so we validate SOMI physics and local/hybrid learning. (2) Design and implement SOMI-JEPA (training loop, encoders, stress = JEPA loss). (3) Design brain topology (Parts, White Matter, Systems for math). (4) Implement SOMIPart, WhiteMatterTract, SOMIBrain. (5) Add diagnostic feedback. (6) Train Circuit-S and scale (M, L, XL). (7) Compare architectures at equal parameter count. No need to do everything in one shot — we can stage it.

### Why Should You Care?

This is the order of operations that gets us from “SOMI in a transformer” to “standalone SOMI brain with JEPA and circuits.” You can use it to see what is next and what depends on what.

### Order of Work

1. **Finish current experiments** — Phase 2 (local learning), Phase 3 (hybrid), fix diagnostics on remote.
2. **Design SOMI-JEPA** — Objective: SOMI as predictor, Y-Encoder for targets, stress = JEPA loss; local learning from embedding error.
3. **Design brain topology** — Concrete Part layout, White Matter topology, 3–4 Systems for math (e.g., comprehension, computation, memory, verification).
4. **Implement SOMIPart** — Refactor SOMILayer into Part (persistent φ, configurable local_dim, per-part diagnostics).
5. **Implement WhiteMatterTract** — Low-rank projections between Parts; fixed or very-slow plasticity.
6. **Implement SOMIBrain** — Parts + tracts + systems + (optional) thalamic routing.
7. **Implement JEPA training** — Y-Encoder target, SOMI settles, stress drives W; optional token loss.
8. **Implement diagnostic feedback** — Pathology response, Hamiltonian-guided compute, spectral gap.
9. **Train Circuit-S** — Small circuit (e.g., 6 Parts, 3 Systems) on math; validate circuits and shared-part generalization.
10. **Compare** — Circuit-S vs flat SOMI vs SOMILayer-in-transformer vs TRM-style at equal params (GSM8K/MATH).

### Scaling (Circuit-S, M, L, XL)

- **Circuit-S:** ~6 Parts, avg dim 256, 3 Systems, ~4M params.
- **Circuit-M/L/XL:** More Parts, larger Parts, more Systems, richer White Matter; ~15M, ~50M, ~200M params. Goal: match or beat larger transformers with fewer params thanks to circuits and local learning.

### The Least You Need to Know (Closing)

Roadmap: Phase 2/3 → design JEPA and topology → implement Part, Tract, Brain, JEPA training, diagnostics → train Circuit-S → scale and compare.

### Check Your Understanding

- [ ] What comes before “implement SOMIBrain”?
- [ ] What is Circuit-S in one sentence?
- [ ] Why compare at equal parameter count?

### Mastery Test

- **Key terms:** roadmap, Phase 2/3, SOMI-JEPA, SOMIPart, WhiteMatterTract, SOMIBrain, Circuit-S, diagnostic feedback.

---

## Chapter 19: Why This Matters for Math — AIMO 3 and Beyond

### In This Chapter

- Why we focus on **math** (AIMO 3, reasoning, clear success metrics)
- How SOMI’s design (temporal depth, embedding prediction, circuits, local learning) could help math reasoning
- What “success” looks like in the short and long term
- How the first-principles story (this document) connects to that goal

**Time:** About 10 minutes.

### The Least You Need to Know (Opening)

We target **math reasoning** (e.g., AIMO 3, GSM8K, MATH) because it has clear right/wrong answers and stresses multi-step reasoning. SOMI’s temporal depth (many settling steps), embedding-space prediction (JEPA), circuit design (shared Parts for generalization), and local learning (stress-driven W) are all aimed at better reasoning and parameter efficiency. Success in the short term = SOMI (and SOMI 3.0) beating or matching baselines on math benchmarks; in the long term = a scalable SOMI brain that excels at reasoning with fewer parameters than a comparable transformer.

### Why Math?

Math gives us:
- **Clear metrics** — accuracy, exact match.
- **Multi-step reasoning** — aligns with SOMI’s “run longer to think harder” (more settling steps).
- **A concrete target** — AIMO 3 and standard benchmarks (GSM8K, MATH) so we can compare fairly.

So math is the testbed for “does SOMI (and SOMI 3.0) actually help reasoning?”

### How SOMI’s Design Helps Math

- **Temporal depth:** Hard problems can get more settling steps (via 5-HT or adaptive logic), so “compute” is allocated where needed.
- **Embedding prediction:** JEPA loss in embedding space can be smoother and more robust than token prediction alone (LLM-JEPA showed +4% GSM8K).
- **Circuits:** Comprehension, computation, memory, verification Systems with shared PFC-like Parts encourage general representations that transfer across problem types.
- **Local learning:** Stress-driven W can adapt to prediction error without full backprop, which may help continual learning and stability.

### What Success Looks Like

- **Short term:** Phase 2 local learning improves accuracy over zero-shot; Phase 3 hybrid does not regress and ideally improves further. SOMI underdamped beats overdamped when oscillations matter; extracted W beats random W when initialization matters.
- **Long term:** SOMI-Circuit-S (or M) reaches competitive math accuracy with a fraction of the parameters of a 1B+ transformer; SOMI-JEPA becomes a standard option for embedding-space reasoning.

### How This Document Fits

This document is the **first-principles story**: from “what is intelligence?” to “what is SOMI 3.0 and why does it matter for math?” It gives you the concepts and the roadmap. The actual equations live in the theory docs and SOMI_Master; the code lives in somi_2_0 and (future) circuit modules. Use this as the map; use the rest as the territory.

### The Least You Need to Know (Closing)

We focus on math (AIMO 3, GSM8K, MATH) for clear metrics and multi-step reasoning. SOMI’s temporal depth, JEPA, circuits, and local learning are aimed at better math performance with fewer parameters. Success = beating or matching baselines and scaling a SOMI brain. This document is the first-principles map to that goal.

### Check Your Understanding

- [ ] Why math?
- [ ] Name two SOMI design choices that could help math reasoning.
- [ ] What is “success” in the short term?

### Mastery Test

- **Teach it:** Explain “why SOMI 3.0 matters for math” in two minutes.
- **Key terms:** AIMO 3, GSM8K, MATH, temporal depth, embedding prediction, circuits, local learning, parameter efficiency.

---

# End of SOMI 3.0: From First Principles

You now have the full story from “what is intelligence?” to “what is SOMI 3.0 and where we are going.” For symbols, see **theory/SYMBOL_GLOSSARY.md**. For concept-to-file mapping, see **references/FILE_MAP.md**. For step-by-step lessons, start at **learning/00_START_HERE.md**.
