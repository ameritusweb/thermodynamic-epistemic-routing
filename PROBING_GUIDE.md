# Identifying and Overcoming Spurious Correlates in Neural Network Probing

## The Core Problem

Gradient descent optimizes for **correlation with labels**, not **causal proximity to labels**.
It has no mechanism to distinguish between:

- A feature that *caused* the label
- A feature that *correlates with* the cause of the label

Both look identical to the loss function. The optimizer finds whichever is easier to fit first,
and once accuracy is high enough, the gradient pressure to keep searching disappears.

This produces probes that report high accuracy while measuring the wrong thing — quietly,
without obvious failure signals. The probe doesn't fail. It succeeds at the wrong task.

---

## A Taxonomy of Signal Types

Understanding what kind of signal you're looking for determines which techniques apply.

**Spatial signals (state-based)**
The information is encoded as the geometric position of a representation in embedding space
at a specific layer. A linear probe or MLP reading activations is appropriate when the signal
is genuinely spatial.

**Dynamic signals (process-based)**
The information is encoded in *how the representation moves* through space across layers —
the trajectory, not the endpoint. Standard probes applied to hidden states will read a
downstream echo of this signal, not the signal itself.

**Direct signals**
The feature you want is present in the input to the probe.

**Echo signals**
The feature you want is a transformed, downstream consequence of the true feature.
It is present and learnable, but it is not the source. A probe reading an echo will
generalize less, be more brittle, and systematically underestimate the true separability
of the classes.

### The Key Diagnostic

> If a zero-parameter probe outperforms a trained probe, you are reading an echo.

The relationship between zero-parameter and trained probe accuracy is itself informative:

| Relationship | Interpretation |
|---|---|
| Trained ≈ zero-parameter | Signal is clean; probe is appropriate |
| Trained >> zero-parameter | Signal is complex; probe adds genuine value |
| **Zero-parameter >> trained** | **You are reading an echo; change approach** |

---

## Detection Methods

### 1. The Zero-Parameter Test

Compute the simplest possible encoding of your theoretical prior — a threshold, a norm,
a ratio — and compare its accuracy to a trained probe on the same data.

This test requires a prior: you must have some hypothesis about what the true signal is.
If you do, compute it directly and compare. A trained probe that cannot beat your hand-coded
formula is reading the wrong feature.

*Example from this project:* T(x) = mean(||h_{n+1} − h_n||) at layers 16–19 achieved 98.39%
accuracy. A trained MLP on concatenated layer activations achieved 89.51%. The 9pp gap
revealed the MLP was reading spatial echoes of a dynamic signal.

### 2. Distribution Shift Test

Train your probe on one data distribution and evaluate on another where the spurious correlate
is unlikely to hold — different prompt styles, different domains, adversarially constructed
examples. A probe reading the true signal should generalize; one reading a spurious echo
typically collapses.

This test does not require a prior. It only requires out-of-distribution data.

### 3. Causal Intervention Test

Use activation patching: swap the activations of a factual example into a speculative
forward pass at the layer you believe carries the signal, and measure whether the probe's
output changes as expected. A spurious probe often fails to respond correctly to
interventions on the true feature because it is sensitive to the echo, not the cause.

### 4. Input Ablation

Explicitly remove the suspected spurious feature from the probe's input. If accuracy drops
to near chance, the probe was reading that feature entirely. If accuracy is maintained, the
probe has access to something else — potentially the true signal.

### 5. Layer Sweep

Train the same probe architecture at every layer. If accuracy peaks at a layer far from
where the signal should be (e.g., peaks at the final layer when you expect mid-network
dynamics), the probe may be reading a propagated echo rather than the source.

---

## Techniques to Overcome Spurious Correlates

### Technique 1: Representation Design

The most reliable fix. Compute the feature you actually want and give it to the probe
directly. If you want differences between layers, compute the differences before the probe
sees them. If you want norms, compute the norms. Do not ask gradient descent to rediscover
arithmetic you already know.

The optimizer has no incentive to learn a complex two-step computation (subtract, then norm)
when a simpler linear correlation achieves acceptable loss. The moment you hand it the
computation's output, that incentive problem disappears.

*When to use:* When you have theoretical knowledge of what the true signal is.

### Technique 2: Thermodynamic / Dynamic Probing

Instead of asking "where is this representation?" ask "how did it get there?"

Compute statistics of the trajectory across layers rather than the state at any single layer:

- **Delta magnitude:** `T(x) = mean(||h_{n+1} − h_n||)` across targeted layers
- **Delta direction:** cosine similarity between consecutive deltas (trajectory curvature)
- **Trajectory length:** cumulative norm across all layers
- **Anisotropy:** variance in delta direction across layers

These are zero-parameter probes that read the *process*, not the *state*. They are
appropriate when the training signal shaped the *dynamics* of the forward pass rather
than the *geometry* of a specific layer's activation space.

**The thermodynamic signal is reliable when:**
- Training included dynamic constraints (delta magnitude margins, turbulence losses)
- The signal spans multiple layers rather than concentrating at one
- You suspect specialization has occurred (different computational effort per query type)

**Calibration note:** Absolute T(x) values are model- and distribution-specific.
The reliable quantity is the *gap between classes*, not the absolute value.
Always calibrate thresholds on a held-out set; do not hardcode them.

**Why it beats spatial probes when training was dynamic:**
A spatial probe at layer L reads the accumulated result of all computation up to L.
T(x) reads the computation itself — the layers that were directly shaped by the training
objective. The spatial probe is reading a transformed, diluted consequence; T(x) is
reading the primary output of training.

*When to use:* When training objectives included dynamic constraints, or when the true
signal is about computational effort rather than representational content.

### Technique 3: Architecture Constraints

Build the computation you want into the probe architecture. If you are looking for
similarity between representations, use a Siamese network with explicit subtraction.
If you are looking for trajectory properties, process hidden states sequentially rather
than concatenating them.

A standard MLP treats every input dimension as independent and has no inductive bias
toward computing differences. An architecture with explicit difference layers forces the
probe toward the right computation even without prior knowledge of the exact formula.

*When to use:* When you have a prior about the signal's structure but want the network
to refine the computation rather than hand-engineer it entirely.

### Technique 4: Contrastive / Minimal Pair Training

Train on paired examples that differ only in the signal of interest, constructed from
identical contexts. Because context is held constant, the probe cannot exploit any
context-level spurious correlate — it must find the feature that actually differs.

This is the most direct suppression of context-level echoes. It is expensive to collect
but highly reliable. The PairedSampler pattern used in adversarial co-training is an
example: by pairing factual and speculative answers to the same question, all
context-specific features cancel, and only the epistemic signal remains.

*When to use:* When you can construct or collect minimal pairs. Directly eliminates
context-level spurious correlates without requiring prior knowledge of their structure.

### Technique 5: Adversarial Feature Removal

Train an adversary to predict the suspected spurious feature from the probe's
intermediate representation. Add a penalty for making the adversary's task easy.
The probe is forced into a representation where the spurious feature is not decodable.

More complex to implement, but does not require architectural changes and does not
require paired data — only a definition of what you want to remove.

*When to use:* When the spurious feature can be defined clearly enough to train an
adversary for it, and when paired data collection is infeasible.

---

## The Diagnostic Triangle

When evaluating a probe, ask three questions in order:

**1. Does a zero-parameter encoding of your theoretical prior outperform the trained probe?**
If yes: the probe is reading an echo. Change the input representation or switch to a
dynamic probe. The trained probe is not adding value; it found a shortcut.

**2. Does accuracy hold under distribution shift?**
If no: the probe learned a brittle correlate specific to the training distribution.
Use contrastive pairs or adversarial feature removal to suppress it.

**3. Is accuracy consistent with what causal interventions predict?**
If no: the probe is sensitive to something other than the true causal feature.
Use layer sweeps and input ablation to locate the actual source.

A probe that passes all three is reading a genuine, stable, causally relevant signal.
A probe that fails any one of them is reporting a number that is not what it appears to be.

---

## The Underlying Principle

Every technique above is a different way of expressing the same constraint:

> **Remove the probe's access to the wrong signal, or directly encode the right one.**

The spurious correlate exists because gradient descent found a real correlation that
satisfied the loss before finding the true signal. The solution is always one of:

1. Make the true signal more direct to access than the spurious one (representation design,
   dynamic probing)
2. Make the true signal the only available signal (minimal pairs, adversarial removal)
3. Build in structural constraints that make the spurious signal harder to exploit than
   the true one (architecture constraints)

The choice between them depends on what you know, what data you have, and how much
you trust your theoretical prior.

---

## Summary

| Technique | Best For | What It Removes |
|---|---|---|
| Representation design | Known signal structure | Wrong input format |
| Thermodynamic probing | Dynamic / multi-layer signals | State-space echoes of process signals |
| Architecture constraints | Known computation structure | Architectural shortcuts |
| Contrastive / minimal pairs | Context-level correlates | Context-specific spurious features |
| Adversarial feature removal | Identifiable spurious features | Named confounds |
| Zero-parameter test | Detection only | — |
| Distribution shift test | Detection only | — |
| Causal intervention | Detection and attribution | — |
