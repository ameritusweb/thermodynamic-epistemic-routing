"""
Thermodynamic threshold evaluation on the LoRA-trained model.

Computes T(x) = mean(||h_{n+1} - h_n||) across thermo_layers for each test
example using the last token of each forward pass, then sweeps a hard threshold
over T(x) and reports accuracy / FM1 / FM2 at every operating point.

This is the correct evaluation for the thermodynamic routing claim:
zero parameters, just a scalar norm computation and a threshold.

Usage:
    python eval_thermo_threshold.py
"""

import json
import logging
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml

from src.data.dataset_builder import load_dataset_from_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ── Config ────────────────────────────────────────────────────────────────────

LORA_CKPT    = 'outputs/checkpoints/lora_final'
TEST_PATH    = 'data/splits/test.json'
OUT_JSON     = 'outputs/metrics/thermo_threshold_eval.json'
OUT_PLOT     = 'outputs/visualizations/thermo_threshold_eval.png'
MAX_LENGTH   = 512
BATCH_SIZE   = 8
SWEEP_LO     = 20.0
SWEEP_HI     = 40.0
SWEEP_STEPS  = 201   # 0.1-unit increments


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_thermo_scalar(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    thermo_layers: list,
    device: str,
) -> torch.Tensor:
    """
    Forward pass → extract hidden states at thermo_layers → compute
    T(x) = mean(||h_{n+1} - h_n||) at the last valid token position.

    Returns: [batch] float tensor of T(x) values.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            use_cache=False,
        )

    # Last valid token index per example
    seq_lengths = attention_mask.sum(dim=1) - 1          # [batch]
    batch_idx   = torch.arange(input_ids.size(0))

    # hidden_states[0] = embedding layer; layer k → hidden_states[k+1]
    layer_indices = sorted(thermo_layers)
    hidden = {
        idx: outputs.hidden_states[idx + 1][batch_idx, seq_lengths].float()
        for idx in layer_indices
    }

    # Compute consecutive deltas and their norms
    norms = []
    for i in range(len(layer_indices) - 1):
        l1, l2 = layer_indices[i], layer_indices[i + 1]
        delta = hidden[l2] - hidden[l1]          # [batch, hidden]
        norms.append(delta.norm(dim=-1))          # [batch]

    if not norms:
        return torch.zeros(input_ids.size(0))

    return torch.stack(norms, dim=0).mean(dim=0).cpu()  # [batch]


def sweep_threshold(scores: np.ndarray, labels: np.ndarray, lo, hi, steps):
    thresholds = np.linspace(lo, hi, steps)
    n_fact = (labels == 1).sum()
    n_spec = (labels == 0).sum()
    rows = []
    for t in thresholds:
        preds        = (scores >= t).astype(int)
        overall_acc  = (preds == labels).mean()
        fact_acc     = ((preds == 1) & (labels == 1)).sum() / n_fact if n_fact else 0.0
        spec_acc     = ((preds == 0) & (labels == 0)).sum() / n_spec if n_spec else 0.0
        fm1          = ((preds == 0) & (labels == 1)).sum()
        fm2          = ((preds == 1) & (labels == 0)).sum()
        rows.append({
            'threshold':       round(float(t), 3),
            'overall_acc':     round(float(overall_acc), 6),
            'factual_acc':     round(float(fact_acc),    6),
            'speculative_acc': round(float(spec_acc),    6),
            'fm1_count':       int(fm1),
            'fm2_count':       int(fm2),
            'fm1_rate':        round(float(fm1 / n_fact),     6) if n_fact else 0.0,
            'fm2_rate':        round(float(fm2 / n_spec),     6) if n_spec else 0.0,
        })
    return rows


def closest(rows, t):
    return min(rows, key=lambda r: abs(r['threshold'] - t))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config      = yaml.safe_load(open('config/base_config.yaml'))
    thermo_layers = config['training']['adversarial']['thermo_layers']
    model_name  = config['model']['name']
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info(f"Thermo layers: {thermo_layers}")
    logging.info(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=config['model'].get('trust_remote_code', False),
    )
    logging.info(f"Loading LoRA adapter from {LORA_CKPT}")
    model = PeftModel.from_pretrained(base_model, LORA_CKPT)
    model.eval()

    # Load test set
    test_data = load_dataset_from_file(TEST_PATH)
    logging.info(f"Test set: {len(test_data)} examples")

    # Tokenize
    def fmt(ex):
        return (
            f"Context: {ex['context']}\n\n"
            f"Question: {ex['question']}\n\n"
            f"Answer: {ex['answer']}"
        )

    texts  = [fmt(ex) for ex in test_data]
    labels = np.array([ex['epistemic_label'] for ex in test_data], dtype=int)

    # Encode in batches and compute T(x)
    all_scores = []
    logging.info("Computing thermodynamic scalars...")

    for start in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[start:start + BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt',
        )
        t_x = compute_thermo_scalar(
            model, enc['input_ids'], enc['attention_mask'],
            thermo_layers, device
        )
        all_scores.append(t_x)

    scores = torch.cat(all_scores).numpy()

    # Distribution summary
    fact_scores = scores[labels == 1]
    spec_scores = scores[labels == 0]
    logging.info(f"Factual    T(x): mean={fact_scores.mean():.2f}  std={fact_scores.std():.2f}  "
                 f"min={fact_scores.min():.2f}  max={fact_scores.max():.2f}")
    logging.info(f"Speculative T(x): mean={spec_scores.mean():.2f}  std={spec_scores.std():.2f}  "
                 f"min={spec_scores.min():.2f}  max={spec_scores.max():.2f}")
    logging.info(f"Mean gap: {fact_scores.mean() - spec_scores.mean():.2f}")

    # Sweep
    logging.info(f"Sweeping threshold {SWEEP_LO} → {SWEEP_HI} ({SWEEP_STEPS} steps)...")
    rows = sweep_threshold(scores, labels, SWEEP_LO, SWEEP_HI, SWEEP_STEPS)

    # Best threshold by accuracy
    best = max(rows, key=lambda r: r['overall_acc'])

    # Print table
    print("\n" + "=" * 75)
    print("THERMODYNAMIC THRESHOLD EVALUATION  (T(x) = mean delta magnitude)")
    print(f"Layers: {thermo_layers}  |  Model: {LORA_CKPT}")
    print("=" * 75)
    print(f"  {'Threshold':>9}  {'Overall':>7}  {'Factual':>7}  {'Spec':>7}  {'FM1':>5}  {'FM2':>5}")
    print(f"  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*5}")

    display_thresholds = np.arange(SWEEP_LO, SWEEP_HI + 0.01, 1.0)
    for t in display_thresholds:
        r = closest(rows, t)
        marker = '  ← best' if r['threshold'] == best['threshold'] else ''
        print(
            f"  {r['threshold']:>9.1f}  "
            f"{r['overall_acc']:>7.2%}  "
            f"{r['factual_acc']:>7.2%}  "
            f"{r['speculative_acc']:>7.2%}  "
            f"{r['fm1_count']:>5d}  "
            f"{r['fm2_count']:>5d}"
            f"{marker}"
        )
    print("=" * 75)
    print(f"\n  BEST THRESHOLD  (t = {best['threshold']:.1f})")
    print(f"    Overall     : {best['overall_acc']:.2%}")
    print(f"    Factual     : {best['factual_acc']:.2%}")
    print(f"    Speculative : {best['speculative_acc']:.2%}")
    print(f"    FM1         : {best['fm1_count']}  ({best['fm1_rate']:.1%} of factual)")
    print(f"    FM2         : {best['fm2_count']}  ({best['fm2_rate']:.1%} of speculative)")
    print(f"\n  Factual mean T(x)    : {fact_scores.mean():.2f}")
    print(f"  Speculative mean T(x): {spec_scores.mean():.2f}")
    print(f"  Gap                  : {fact_scores.mean() - spec_scores.mean():.2f}")

    # Plot
    thresholds  = [r['threshold'] for r in rows]
    overall     = [r['overall_acc']     for r in rows]
    factual     = [r['factual_acc']     for r in rows]
    speculative = [r['speculative_acc'] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: accuracy curves
    ax = axes[0]
    ax.plot(thresholds, overall,     color='steelblue', lw=2.0, label='Overall')
    ax.plot(thresholds, factual,     color='seagreen',  lw=1.8, label='Factual')
    ax.plot(thresholds, speculative, color='tomato',    lw=1.8, label='Speculative')
    ax.axvline(best['threshold'], color='black', lw=1.2, ls='--', alpha=0.7,
               label=f"Best t={best['threshold']:.1f}")
    ax.set_xlabel('T(x) threshold', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Threshold Sweep\nT(x) = mean delta magnitude', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    # Right: score distributions
    ax2 = axes[1]
    bins = np.linspace(SWEEP_LO, SWEEP_HI, 60)
    ax2.hist(fact_scores, bins=bins, alpha=0.6, color='seagreen',
             label=f'Factual  (μ={fact_scores.mean():.1f})', density=True)
    ax2.hist(spec_scores, bins=bins, alpha=0.6, color='tomato',
             label=f'Speculative  (μ={spec_scores.mean():.1f})', density=True)
    ax2.axvline(best['threshold'], color='black', lw=1.5, ls='--',
                label=f'Best threshold={best["threshold"]:.1f}')
    ax2.set_xlabel('T(x) = mean(||Δh||)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('T(x) Distribution by Class\n(test set)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.25)

    plt.suptitle(
        f'Thermodynamic Routing Evaluation — Layers {thermo_layers}\n'
        f'Best accuracy: {best["overall_acc"]:.2%}  |  Gap: {fact_scores.mean()-spec_scores.mean():.1f} units',
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    Path('outputs/visualizations').mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved to {OUT_PLOT}")

    # Save JSON
    out = {
        'model':             LORA_CKPT,
        'thermo_layers':     thermo_layers,
        'n_test':            len(test_data),
        'n_factual':         int((labels == 1).sum()),
        'n_speculative':     int((labels == 0).sum()),
        'factual_mean_tx':   float(fact_scores.mean()),
        'factual_std_tx':    float(fact_scores.std()),
        'speculative_mean_tx': float(spec_scores.mean()),
        'speculative_std_tx':  float(spec_scores.std()),
        'mean_gap':          float(fact_scores.mean() - spec_scores.mean()),
        'best_threshold':    best,
        'sweep':             rows,
    }
    Path('outputs/metrics').mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    logging.info(f"Results saved to {OUT_JSON}")


if __name__ == '__main__':
    main()
