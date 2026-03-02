# Thermodynamic Epistemic Routing

**98.39% factual/speculative classification accuracy. Zero additional parameters at inference.**

A training method that forces a language model to develop distinct computational dynamics for
factual retrieval vs. speculative generation — then reads that signal directly from the
physics of the forward pass.

---

## The Core Finding

Standard interpretability probes ask: *where is this representation in embedding space?*

This project asks: *how hard did the model work to get there?*

Factual retrieval leaves a high-energy signature in the model's hidden state trajectory.
Speculative generation is smooth — the model glides. After training with a thermodynamic
loss, the gap between these two regimes is 13.62 units. A single threshold classifies them
at 98.39% accuracy with no additional parameters.

```
T(x) = mean(||h_{n+1} − h_n||)   for layers 16–19, at last token

T(x) > 27.2  →  factual regime
T(x) < 27.2  →  speculative regime
```

For comparison, a trained MLP probe on the same model's activations achieves 89.51%.
The 9pp gap is not a failure of the MLP — it's evidence that the training objective
shaped *dynamics*, not geometry. The MLP was reading the downstream echo; T(x) reads the source.

---

## Results

| Method | Accuracy | Factual | Speculative | FM2 | Params |
|--------|:--------:|:-------:|:-----------:|:---:|:------:|
| Baseline MLP (frozen model) | 85.9% | ~85% | ~87% | — | 953K |
| Best spatial MLP (post-LoRA) | 91.15% | 93.7% | 88.6% | — | 953K |
| **T(x) threshold (zero-param)** | **98.39%** | **99.31%** | **97.31%** | **2.7%** | **0** |

FM2 = speculative responses misclassified as factual (the dangerous failure mode).

![Thermodynamic threshold evaluation — T(x) distributions and accuracy sweep](outputs/visualizations/thermo_threshold_eval.png)

Full experimental record: [RESULTS.md](RESULTS.md)
Research narrative: [DISCOVERY.md](DISCOVERY.md)

---

## How It Works

### Training

The model (Qwen2.5-1.5B + LoRA) is trained with `ThermoSpatialLoss`:

```
L_thermo  = mean(relu(margin − (mag_fact − mag_spec)))   per same-context pair
L_spatial = relu(spatial_margin − ||c_fact − c_spec||_2)  centroid L2 at deepest layer

L_routing = L_BCE + λ_thermo × L_thermo + λ_contrastive × L_spatial
L_total   = L_generation + λ_routing × L_routing
```

Training uses a `PairedSampler` that interleaves `(factual, speculative)` pairs from the
same context in each batch. This context-normalises the magnitude gap — the constraint is
on the *difference within a pair*, not the absolute magnitude.

An adversarial co-evolution loop alternates:
- **Every step**: generator update — routing loss gradient flows through activations to LoRA
- **Every 5 steps** (after 100-step warmup): predictor update on detached activations

### What Happened During Training

By epoch 3, the constraint was trivially satisfied (`loss_thermo ≈ 0`). The model continued
widening the gap through epoch 6 anyway — it had internalised the specialisation as the
optimal generation strategy, not just a loss target.

The generation loss *decreased* as the gap widened: 1.914 → 1.835. Uniform computation
was a free local optimum under cross-entropy alone. The thermodynamic constraint made that
optimum expensive. The model found that specialisation reduces generation loss as well.

| Metric | Baseline | Epoch 3 | Epoch 6 |
|--------|:--------:|:-------:|:-------:|
| Factual magnitude | ~26 | ~32 | **36.22** |
| Speculative magnitude | ~22.5 | ~22 | **22.59** |
| Gap | ~3.5 | ~10 | **13.62** |
| Eval LM loss | — | 1.842 | **1.835** |

### Inference (Zero Parameters)

```python
# Inside your generation loop — 4 lines
T = mean(norm(h[17] - h[16]), norm(h[18] - h[17]), norm(h[19] - h[18]))
if T < 27.2:
    # speculative regime — trigger grounding feedback
    handle_speculation()
```

No MLP, no extra model call, no latency. Just norm computations on hidden states already
computed during the forward pass.

---

## Live Demo

`generate_with_epistemic_routing.py` streams output token-by-token with colors:
- **Green**: factual regime (T(x) > threshold)
- **Red**: speculative regime (T(x) < threshold)

```bash
python generate_with_epistemic_routing.py
```

You will see the model switch regimes mid-sentence when it moves from retrieval to confabulation.

---

## Setup

### Requirements

- GPU with ≥24GB VRAM (A100/H100/Blackwell recommended for training; inference runs on less)
- Python 3.10+
- CUDA 12.1+

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env (required for dataset generation only)
```

### Reproducing the Full Pipeline

```bash
# 1. Generate oracle dataset (~$50 API cost, ~5 hours)
python main.py --phase data

# 2. Train baseline predictor on frozen model
python main.py --phase predictor

# 3. LoRA fine-tuning with ThermoSpatialLoss (~12 hours on A100)
python main.py --phase lora

# 4. Zero-parameter threshold evaluation (the headline result)
python eval_thermo_threshold.py

# 5. Topology evaluation
python main.py --phase eval
```

### Using Pre-Trained Weights

If you want to skip training and run the demo or threshold evaluation:

1. Download the LoRA adapter from [HuggingFace — link TBD]
2. Place it at `outputs/checkpoints/lora_final/`
3. Run `python eval_thermo_threshold.py` or `python generate_with_epistemic_routing.py`

### Docker (RunPod)

```bash
docker build -f docker/Dockerfile -t epistemic-routing .
# Or use the RunPod entrypoint directly:
bash docker/runpod_entrypoint.sh
```

---

## Project Structure

```
├── eval_thermo_threshold.py        # Zero-parameter threshold sweep (headline eval)
├── generate_with_epistemic_routing.py  # Live color-coded generation demo
├── main.py                         # Full pipeline orchestration
├── layer_sweep.py                  # Layer-by-layer signal analysis
├── train_post_lora_predictor.py    # Post-LoRA MLP probe (comparison baseline)
├── train_multi_feature_predictor.py
├── clean_dataset.py                # Refusal-leakage cleaning
├── generate_augmentation.py        # Hard-negative / complex-factual augmentation
├── analyze_errors.py               # Error analysis
├── config/
│   └── base_config.yaml            # All hyperparameters
├── src/
│   ├── data/
│   │   ├── dataset_builder.py
│   │   ├── paired_sampler.py       # Same-context (factual, speculative) pair batching
│   │   └── question_generator.py
│   ├── models/
│   │   ├── predictor.py            # MLP classifier
│   │   ├── activation_extractor.py
│   │   └── multi_feature_extractor.py
│   ├── training/
│   │   ├── custom_trainer.py       # EpistemicRoutingTrainer
│   │   ├── thermo_spatial_loss.py  # ThermoSpatialLoss
│   │   ├── phase1_predictor.py
│   │   └── phase2_lora.py          # Adversarial alternating loop
│   └── evaluation/
│       ├── metrics_calculator.py
│       └── topology_visualizer.py
├── outputs/
│   └── metrics/                    # All result JSON files
├── DISCOVERY.md                    # How we got here — full research narrative
├── RESULTS.md                      # Complete experimental record
├── PROBING_GUIDE.md                # General guide: spurious correlates in probing
├── ERROR_ANALYSIS.md               # Error analysis and augmentation experiment
└── whitepaper.md                   # Original theoretical framework
```

---

## Why This Matters

Standard mechanistic interpretability uses probes trained with gradient descent on
activations. If the signal is *dynamic* (a property of the forward pass trajectory) rather
than *spatial* (a property of activation geometry at a single layer), those probes will
systematically find the wrong thing and report high accuracy doing it.

The 9-point gap between T(x) (98.39%) and the best MLP (91.15%) on the same model is a
direct measurement of that failure mode. The MLP found a high-quality spatial echo of the
true signal and gradient descent had no incentive to look further.

See [PROBING_GUIDE.md](PROBING_GUIDE.md) for a general treatment of when this failure mode
occurs and how to detect and correct it.

---

## Practical Application: AI Swarm Auto-Feedback

The original motivation for this project: detect when an agent in a multi-agent system
slips into speculation without having grounding data in context, and trigger corrective
feedback before downstream agents consume the ungrounded output.

At 2.7% FM2, approximately 97.3% of speculative outputs are correctly flagged. A swarm
of 50 agents each generating 500 tokens would have ~67 false negatives across the entire
run — trivially handled by a lightweight auto-feedback rule.

Cost: four norm computations per forward pass. Already in the graph. Zero extra latency.

---

## Citation

If you use this work, please cite:

```
@misc{epistemic-routing-2026,
  title  = {Thermodynamic Epistemic Routing: Zero-Parameter Hallucination Detection
             via Forward-Pass Energy},
  year   = {2026},
  url    = {https://github.com/ameritusweb/thermodynamic-epistemic-routing}
}
```

---

## License

Apache-2.0
