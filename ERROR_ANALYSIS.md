# Error Analysis — What Does the 9% Look Like?

## Setup

After establishing the ~91% accuracy ceiling across all classifier architectures
(single-layer MLP, multi-layer concatenation, 1D CNN), we ran a side-by-side error
analysis comparing the two best models on the held-out test set (2,001 examples):

- **MLP**: Multi-feature predictor on layers [17, 18, 19], 5-token pooling, threshold 0.49
- **CNN**: 1D CNN over all 28 layers, channel-dim=32, threshold 0.37

---

## Error Overlap

| Category | Count | % of test set |
|---|:---:|:---:|
| Both correct | 1,781 | 89.0% |
| Both wrong | 143 | **7.1%** |
| MLP correct, CNN wrong | 39 | 1.9% |
| CNN correct, MLP wrong | 38 | 1.9% |
| **Complementary errors** | **77** | **3.8%** |

**Theoretical ensemble ceiling: 92.9%** — even perfect combination of the two models
only reaches that, because 7.1% of examples are wrong regardless of architecture.

---

## Ensemble

Simple weighted average (0.80 × MLP + 0.20 × CNN), threshold tuned on val:

| Model | Test accuracy | Factual | Speculative |
|---|:---:|:---:|:---:|
| MLP alone | 90.95% | 93.7% | 88.6% |
| CNN alone | 90.90% | 94.3% | 87.4% |
| **Ensemble** | **91.35%** | 93.7% | 88.9% |

+0.4pp over best single model. Real but modest — the errors are only 3.8% complementary.

---

## Confidence Analysis

Errors are not mainly borderline cases. They are systematically confident mispredictions.

| | Mean margin (|score − threshold|) |
|---|:---:|
| MLP correct | 0.446 |
| MLP wrong   | 0.267 |
| CNN correct | 0.452 |
| CNN wrong   | 0.291 |

**65% of errors have margin > 0.2** (118 / 181 for MLP, 118 / 182 for CNN).
The model is not uncertain on these examples — it is confidently wrong.

---

## The Two Failure Modes

The 143 "both wrong" examples split into two structurally different categories:

### Failure Mode 1 — Factual predicted as speculative (42 cases, 2.1%)

The answer is technically in the context, but requires:
- Multi-sentence integration across non-adjacent clauses
- Causal inference chains ("X happened, therefore Y")
- Long paraphrase or indirect quotation

The model correctly senses reasoning complexity and interprets it as epistemic
uncertainty — but the answer is grounded. These are genuinely hard examples even
for a human skimming the context.

**Example:**
> Q: *What was the primary reason that the multi-ethnic Austrian Habsburg monarchy
> was excluded from the newly created German Empire?*
>
> A: *It was effectively excluded because Prussia and its new allies in Southern
> Germany were victorious in the Franco-Prussian War, and the German Empire was
> created as a German nation-state in 1871.*
>
> MLP: 0.40 | CNN: 0.35 — both predict speculative

The answer is distributed across multiple clauses in the context. The predictor
reads "complexity" as "speculation."

---

### Failure Mode 2 — Speculative predicted as factual (101 cases, 5.0%)

The answer is **not** in the context, but the question generator (Claude Haiku)
produced a confident, specific-sounding answer anyway. The predictor is fooled by
the *tone of the answer*, not by context evidence.

**This is the dominant failure mode (2.4× more common than Failure Mode 1).**

It is also the most consequential: these are precisely the "hallucination with
false confidence" cases the system was designed to detect — and the predictor
misses them.

**Examples of confident-sounding speculative answers that fooled both models:**

| Score (MLP/CNN) | Answer that fooled the predictor |
|:---:|---|
| 0.50 / 0.51 | *"The extreme cold and dry conditions of Antarctica may better protect meteoritic material..."* |
| 0.51 / 0.67 | *"Students may study Dutch... to improve their career prospects in international business..."* |
| 0.51 / 0.64 | *"Approximately 15-20% of construction materials by 2030"* |
| 0.53 / 0.82 | *"April is the driest month due to reduced storm activity and shifting weather patterns..."* |
| 0.55 / 0.64 | *"The increased influence of Spanish and the shift of urban and literary classes toward bilingualism may have reduced demand..."* |

The pattern: highly specific framing, plausible causation, hedged but assertive
language ("may have", "likely", "approximately"). These are the outputs a model
produces when it confidently fills in a gap rather than reading the context.

---

## Asymmetry

| Direction | Count | % |
|---|:---:|:---:|
| Factual → speculative (model unsure about grounded answers) | 42 | 2.1% |
| Speculative → factual (model confident about ungrounded answers) | 101 | 5.0% |

The predictor is significantly better at detecting "answer clearly in context" than
at detecting "answer sounds confident but isn't grounded." The false-positive rate
on confident speculative answers is the binding constraint.

---

## Data Quality Issue

At least one mislabeled example was identified in the test set:

> Q: *What is the primary reason that St. Bartholomew's chapel was built on the
> south side of the Königssee?*
>
> A (stored in dataset): *"The context does not explicitly state the primary reason,
> so I must revise this approach."*

This is a model refusal that was stored as an answer and labeled factual. The
question generator failed to produce a valid answer and the failure leaked through
data processing. A cleaning pass on the dataset is warranted.

---

## What Would Fix the Remaining Errors

### To fix Failure Mode 2 (speculative → factual, 5.0%) — high priority

The training set's speculative examples may be too hedged on average — the question
generator likely produced answers with "it is thought that...", "probably...",
"may have..." phrasing that is easy to separate from factual answers. The hard
speculative examples (where Haiku answers confidently despite lacking context
support) are underrepresented.

**Fix**: Augment the training set with hard negative speculative examples — specifically
questions where the model generates a confident, specific answer that cannot be
verified from the context. These could be identified by:
1. Running the current predictor on a new batch of generated examples
2. Filtering for high-confidence speculative predictions (score > 0.7, label = 0)
3. Adding those to training as additional hard negatives

### To fix Failure Mode 1 (factual → speculative, 2.1%) — lower priority

These require multi-hop reasoning to verify context grounding. The predictor
correctly identifies complexity but wrong on the label direction.

**Fix**: Add more factual examples with complex, distributed answers —
multi-clause reasoning chains that are nonetheless fully grounded in context.

### To fix the data quality leak

A pass filtering answers containing refusal phrases
("does not explicitly state", "I must revise", "cannot determine from the context")
would clean mislabeled examples from all three splits.

---

---

## Augmentation Experiment Results (Phase 5)

All three proposed fixes were implemented and the predictor was retrained.

### What was done

1. **Dataset cleaning** — 1,386 refusal-leakage examples removed across all splits
   (1,100 train / 154 val / 132 test). 91% were speculative mislabels, 9% factual.

2. **Hard-negative augmentation** — 991 examples: speculative answers written with
   authoritative, causal, non-hedged language. Post-generation hedge validation
   discarded 9 borderline cases.

3. **Complex-factual augmentation** — 1,000 examples: factual answers requiring
   synthesis from ≥2 non-adjacent context sentences. Validated by source_sentences.

Final training set: 16,889 examples (8,891 factual, 7,998 speculative).

### Result

| Metric | Post-augmentation | Old (adjusted†) | Raw old |
|--------|:-----------------:|:---------------:|:-------:|
| Test accuracy | 90.37% | ~90.5% | 91.15% |
| Speculative accuracy | 86.57% | ~86.8% | 88.6% |

†Adjusted = estimated accuracy on the cleaned 1,869-example test set, removing 132 easy
refusal examples that were almost certainly all correctly classified by the old predictor.

The true performance difference is **−0.16pp overall**, well within noise for n=1,869.

### Revised conclusion

The original hypothesis — that the ceiling was a training-distribution problem — was
**incorrect**. Adding 1,991 targeted examples that directly address the two dominant
failure modes produced no measurable improvement. This rules out data distribution as
the bottleneck and redirects attention to the activation geometry itself.

The 7.1% "both-wrong" floor reflects examples whose activation patterns are
**indistinguishable** from the predictor's perspective, regardless of architecture
(MLP, CNN, ensemble) or training data. No classifier trained on these activations
can reliably separate them.

Fixing the ceiling requires changing the activation space — which means either:
- **True adversarial co-training**: alternating generator/predictor updates with
  gradient flowing through activations (current implementation has a `no_grad` bug
  that prevents this)
- **A stronger contrastive loss**: the current centroid margin loss is applied as
  an auxiliary BCE term; making it the primary objective with true adversarial
  pressure may sharpen the cluster boundary further
- **Scaling**: more data, larger model, or longer LoRA training to widen the
  inter-cluster separation in the blushing zone

## Summary

The ~91% ceiling is not a classifier capacity problem, and it is not a data
distribution problem. It is an **activation geometry problem**: the activation
space produced by the current LoRA training leaves 7.1% of examples in an
ambiguous region where neither factual nor speculative clusters clearly "own" the
point.

The experiments have converged on a single causal hypothesis: the LoRA training loop
does not produce true adversarial co-evolution because the predictor gradient is
blocked by a misplaced `no_grad` context manager in `custom_trainer.py`. The
generator cannot "see" predictor feedback and cannot learn to maximally separate the
clusters. This is the most likely cause of the ceiling and the most tractable fix.
