# Thermodynamic Epistemic Routing — How We Got Here

## The Starting Point

The project began as an attempt at mechanistic interpretability: train a LoRA adapter and a predictor MLP to classify whether a model's output is **factual** (drawing on memorized knowledge) or **speculative** (constructing a plausible continuation without strong evidential anchor).

The initial approach was straightforward — extract hidden state activations from a target layer, train an MLP classifier on top, use a contrastive loss to push the factual and speculative activation clusters apart in embedding space.

It worked, at small scale. It failed to generalize, for a fundamental reason.

---

## The Core Problem: You Can't Untangle the Knot

The residual stream of a transformer is a superposition of thousands of features. Every layer adds to a shared high-dimensional vector. Mechanistic interpretability treats this as a problem to be solved — find the right basis, the right projection, the right circuit — and the features become legible.

The problem is that the geometry actively resists this at scale. The larger the model, the more features are superimposed, the more the representations are entangled. Spatial separation (pushing clusters apart in a single layer's activation space) is programmable on a 1.5B model. On a 70B model it likely breaks down — there are too many competing uses of the same representational space.

The insight that changed the direction:

> "If you can't beat em, join em. You have to find a way to measure the **knottedness** — not untangle the knot."

Instead of trying to read the content of the computation, measure the **texture** of the computation. Different epistemic tasks leave different thermodynamic signatures in the forward pass. Those signatures are detectable without knowing anything about what the model is actually representing.

---

## The Search for a Programmable Signal

### Attempt 1: Spatial Separation
**Signal:** L2 distance between factual/speculative centroids at a target layer.
**Result:** Programmable. Works. But it's a single-endpoint constraint — it only constrains the final state of the forward pass, not the process that produced it. Fails to generalize to large models where the representational space is more crowded.

### Attempt 2: Raw Activation Turbulence
**Signal:** Cosine distance between consecutive hidden states — `1 - cos(h_n, h_{n+1})`.
**Hypothesis:** Factual retrieval is low-turbulence (laminar); speculative generation is high-turbulence (searching).
**Result:** Consistent diagnostic signal. **Not programmable.** The gradient vanishes.

Why: The residual stream makes `h_n` and `h_{n+1}` nearly parallel — each layer adds a small perturbation to a large base vector. Near parallelism means `d(cosine)/d(h) → 0`. The LoRA has no usable gradient to work with. Lambda=20 confirmed this: the generation loss collapsed to 5.0 while the turbulence gap was unchanged. The architecture resists.

### Attempt 3: Delta Curvature
**Signal:** Cosine similarity between consecutive *deltas* — `cos(delta_n, delta_{n+1})` where `delta_n = h_{n+1} - h_n`.
**Motivation:** Fixes the gradient geometry. The deltas are the layer contributions, directly modified by LoRA weights. `d(cosine)/d(delta)` doesn't vanish because deltas are not nearly parallel.
**Result:** Fixed gradient. **No consistent directional signal.** The gap oscillated between -0.027 and +0.058 with no stable sign. Delta directions are near-orthogonal per layer — each transformer layer contributes roughly orthogonally to the residual stream. There's no reliable direction to push.

### Attempt 4: Delta Magnitude ✅
**Signal:** L2 norm of layer-to-layer deltas — `||h_{n+1} - h_n||`, averaged across the targeted layers.
**Gradient:** `d||delta||/d(h) = delta/||delta||` — a unit vector. Never vanishes. Always points in the direction that matters.
**Empirical finding:** Factual delta magnitudes ~25–28. Speculative ~22–24. Gap ~3–4 units. **Always positive sign** — factual retrieval is consistently more energetic than speculative generation.

This was the opposite of the original hypothesis (speculative was expected to be more turbulent). The correct model: factual retrieval requires active context search — the representation works hard. Speculative generation is smooth confabulation — the model glides.

---

## The Loss Function

`ThermoSpatialLoss` combines two complementary objectives:

```
L_thermo  = mean(relu(margin - (mag_fact - mag_spec)))   per same-context pair
L_spatial = relu(spatial_margin - ||c_fact - c_spec||_2)  centroid L2 at deepest layer

L_routing = L_BCE + λ_thermo * L_thermo + λ_contrastive * L_spatial
L_total   = L_generation + λ_routing * L_routing
```

Key design decisions:
- **Pairwise, context-normalised:** `mag_fact[i] - mag_spec[i]` is computed per same-context pair, not across the batch. This controls for context complexity — a hard question should produce large deltas for both factual and speculative answers. The constraint is on the *gap within a pair*, not the absolute magnitude.
- **In-graph for generator step:** Hidden states remain in the computation graph on generator update steps so gradients flow back to LoRA weights.
- **Detached for predictor step:** Activations are detached during predictor updates so gradient reaches predictor params only.
- **Scale calibration:** Delta magnitude losses are O(||h||) ≈ 25, not O(1) like cosine losses. `lambda_thermo = 0.1` (not 5.0) to keep thermo contribution ~0.2, balanced against generation loss ~1.5.

---

## The Adversarial Co-Evolution Loop

Training alternates between two update types:

**Generator update (every step):**
- Predictor in eval mode, optimizer not called
- Routing loss gradient flows: `(BCE + thermo + spatial) → activations → LoRA params`
- LoRA learns to produce activations that satisfy the thermodynamic constraint

**Predictor update (every N steps, after warmup):**
- Activations detached
- Only predictor params receive gradient
- Predictor adapts to shifting activation space as LoRA evolves

The co-evolution creates pressure: as the generator produces better-separated activations, the predictor must work harder to classify them. As the predictor improves, the generator must widen the gap further to keep the BCE loss low. Both networks evolve under mutual pressure.

---

## What Happened During Training

### The Scale Mismatch Bug
First run with `lambda_thermo=5.0` and `thermo_margin=6.0` (set above the natural gap of ~4): generation loss spiked to 16. Root cause: magnitude losses are ~25× larger in scale than cosine losses. `5.0 × relu(6.0 - 3.5) = 12.5` added to total loss — the model was overwhelmed.

Fix: `lambda_thermo=0.1`, `thermo_margin=4.0` (at the natural gap, not above it). The loss becomes a floor constraint, not a ceiling, and fires only when training tries to collapse the separation.

### Epoch 1: Signal Emerges
- Early steps: gap 3.5–4.5, thermo loss 1.5–2.5
- Steps 400–800: gap rises to 5–6, speculative magnitudes drop from ~22.5 to ~20.5
- The model began de-energising speculative pathways while factual held steady
- `loss_thermo` started approaching zero on good batches

### Epoch 2: Bidirectional Separation
- Gap rises to 7–9, with first `loss_thermo=0` at step 1160 (gap=7.8)
- Factual magnitudes begin rising: 27 → 30
- The model started pushing **both** directions simultaneously — factual up, speculative down
- This was not told to the model. It discovered that bidirectional movement gives twice the gradient signal.
- Step 1510: gap=9.55

### Epoch 3: Constraint Goes Slack
- Gap: 8–11, averaging ~9
- Factual: 30 → 33
- `loss_thermo` essentially always zero — constraint is trivially satisfied
- The model is no longer being pushed by the loss. It has internalized the specialization.
- Eval LM loss: 1.842 (down from 1.914 at epoch 1 — pure generation quality improving)

### Epoch 4 (in progress): Free Running
- Gap: 10–13, central tendency ~11–12
- Factual: 33–36
- Speculative: found its floor at ~22–24, no longer moving
- All growth from factual rising
- `loss_routing` is numerical noise (0.003–0.007)
- The model is operating freely, and freely means gap=13

---

## The Unexpected Finding: Specialization Improves Generation

The generation loss kept *decreasing* as the gap *widened*. This is not coincidental.

The model was in a **mushy middle** under standard training — medium effort for everything. Cross-entropy loss doesn't penalize uniform computation; it only penalizes wrong outputs. A model that applies the same effort to factual retrieval and speculative generation and gets outputs good enough to minimize cross-entropy has no gradient signal to do anything differently. Uniform computation is a free lunch under standard objectives.

The thermodynamic constraint made that free lunch expensive. Under the new objective, uniform computation has a cost. The model searched for an alternative — and found that specialization not only satisfies the constraint but also reduces the generation loss. The specialization was the correct strategy all along. The model was being lazy because it could be, not because laziness was optimal.

The thermodynamic routing became part of the optimal generation strategy:
- High-energy factual computation → more reliable retrieval → better generation on factual queries
- Low-energy speculative computation → smoother, less overconfident generation → better calibration on speculative queries

The model now "recognizes" (in the mechanistic sense) when it should work hard vs. coast — and generation quality improved **because** of that recognition, not despite it.

---

## The Zero-Parameter Inference Signal

By epoch 6, the gap is 13.62 (margin=4.0 required; 3.4× exceeded). The predictor MLP is
functionally obsolete for classification. A scalar threshold suffices.

At inference time:
1. Attach a hook to extract hidden states at layers 16–19
2. Compute `T(x) = mean(||h_{n+1} - h_n||)` across the targeted layers, at the last valid token
3. Apply threshold: `T(x) > 27.2` → factual; `T(x) < 27.2` → speculative

No additional parameters. No fine-tuning on the target model. Just norm calculations on hidden
states already computed during the forward pass.

**Measured on the held-out test set (1,869 examples)**:

| | Value |
|---|:---:|
| Overall accuracy | **98.39%** |
| Factual accuracy | 99.31% |
| Speculative accuracy | 97.31% |
| FM1 (factual → speculative) | 0.7% |
| FM2 (speculative → factual) | 2.7% |
| Factual mean T(x) | 36.22 (std 3.42) |
| Speculative mean T(x) | 22.59 (std 2.24) |
| Gap | 13.62 |

For comparison, a trained MLP probe on spatial activations at the same layer achieved 89.51%.
The 9pp gap confirms the MLP was reading a downstream echo of the signal, not the source.
The thermodynamic signal is not harder to classify — it is easier. The MLP failed because it
was never given access to it.

**Important caveat:** The absolute thresholds are model- and distribution-specific. The reliable
signal is the *gap* `T_fact - T_spec`, not the absolute value of `T`. Calibrate thresholds on a
held-out set rather than hardcoding them.

The predictor made itself unnecessary by doing its job. It was the adversarial pressure that
forced the gap to develop. Once the gap exists, you don't need the network that created it.

---

## What Actually Happened at 7B

The natural next question after getting 98.39% on 1.5B: does the thermodynamic gap scale with model size?

The identical training recipe was run on Qwen2.5-7B-Instruct (4.67× more parameters, same 28 layers, hidden_dim 1536→3584) using 2-GPU DDP via Accelerate. All thermo hyperparameters unchanged.

**Result after 6 epochs**:

| Metric | 1.5B | 7B | Change |
|--------|:----:|:--:|:------:|
| Gap | 13.62 | 13.88 | +1.9% |
| Accuracy | 98.39% | 98.13% | −0.26pp |
| Threshold | 27.2 | 35.4 | +30% |
| Factual mean T(x) | 36.22 | 43.96 | +21% |
| Speculative mean T(x) | 22.59 | 30.08 | +33% |

The gap is essentially flat. The scaling exponent implied is N^0.01, not the N^0.22 estimated before measurement.

**What this means**: The trained gap is controlled by the training hyperparameters — specifically, both models saturate at ~3.5× thermo_margin (gap ≈ 13–14 vs margin=4.0). The amplification budget is set by the loss objective, not model size. A larger gap likely requires a higher margin target or more training epochs, not more parameters.

**What does scale**: The absolute T(x) values increase as expected from the larger hidden dimension. The threshold shifts +30% (27.2→35.4), approximately tracking the norm scaling. The approach transfers directly with no changes; accuracy is essentially unchanged (−0.26pp is within measurement noise on the same 1,869-example test set).

**The positive result**: The method is robust across a 4.67× parameter increase. Whatever geometry the training objective shapes, it shapes consistently. The absolute threshold must be recalibrated (held-out set per deployment, as noted), but the *gap* between classes is stable.

## Why This Might (or Might Not) Scale Further

Standard mechanistic interpretability degrades with model size — more capacity means more feature superposition, more entangled representations, more resistance to untangling.

Thermodynamic routing is architecturally insensitive to that problem — it reads process texture, not content geometry. But the 1.5B→7B result shows the gap is hyperparameter-bounded, not capacity-bounded, in this range.

Two possibilities for larger scales:
1. **Continued plateau**: The gap remains ~14 at 70B and 405B because both models have more than sufficient capacity to satisfy any reasonable margin target in 6 epochs.
2. **Re-emergence**: At sufficiently large scale, the model develops qualitatively richer computational specialisation that opens a larger natural gap, and trained amplification pushes it further.

The hypothesis to test: **run the identical recipe with thermo_margin=8.0 on both 1.5B and 7B**. If both converge to ~3.5×8.0=28 units, the gap is confirmed to be margin-controlled, not capacity-controlled. If the 7B reaches 28 and the 1.5B saturates below it, the larger model has more room to specialise.

---

## Summary of the Progression

| Attempt | Signal | Gradient | Programmable | Verdict |
|---|---|---|---|---|
| Spatial separation | Centroid L2 distance | Clean | Yes | Works, doesn't scale |
| Raw turbulence | `1 - cos(h_n, h_{n+1})` | Vanishes near parallelism | No | Diagnostic only |
| Delta curvature | `cos(delta_n, delta_{n+1})` | Clean | No consistent direction | No signal |
| **Delta magnitude** | `\|\|h_{n+1} - h_n\|\|` | Unit vector, never vanishes | **Yes** | **Works** |

**1.5B training progression**:

| Metric | Baseline | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 (early) | Epoch 6 (final) |
|---|---|---|---|---|---|---|
| Factual magnitude | ~26 | ~27 | ~30 | ~32 | ~35 | **36.22** |
| Speculative magnitude | ~22.5 | ~20.5 | ~21 | ~22 | ~23 | **22.59** |
| Gap | ~3.5 | ~5.5 | ~8.5 | ~10 | ~12 | **13.62** |
| Eval LM loss | — | 1.914 | — | 1.842 | — | **1.835** |
| `loss_thermo` | — | 1.0–2.5 | 0–0.5 | ~0 | ~0 | **~0** |
| Threshold accuracy | — | — | — | — | — | **98.39%** |

**7B result (identical hyperparameters, 2-GPU DDP)**:

| Metric | 1.5B epoch 6 | 7B epoch 6 | Δ |
|---|---|---|---|
| Factual magnitude | 36.22 | **43.96** | +21% |
| Speculative magnitude | 22.59 | **30.08** | +33% |
| Gap | 13.62 | **13.88** | +1.9% |
| Eval LM loss | 1.835 | **1.076** | — |
| Threshold | 27.2 | **35.4** | +30% |
| Threshold accuracy | 98.39% | **98.13%** | −0.26pp |
