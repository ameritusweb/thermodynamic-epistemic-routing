"""
Threshold sensitivity analysis for the epistemic routing MLP.

Sweeps the decision threshold from 0.20 to 0.80 and reports the accuracy /
FM1 / FM2 tradeoff at every point. Produces:

  outputs/visualizations/threshold_analysis.png  — three-curve plot
  outputs/metrics/threshold_analysis.json        — full sweep table

Key operating points highlighted on the plot:
  0.41  — accuracy-optimal (current default, tuned on val)
  0.60  — high-sensitivity (catch near-100% hallucinations)

Usage:
    python threshold_analysis.py
"""

import json
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.predictor import EpistemicPredictor
from src.data.dataset_builder import load_dataset_from_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ── Paths ─────────────────────────────────────────────────────────────────────

MLP_CKPT      = 'outputs/checkpoints/predictor_multi_feat_L17_18_19_T5_best.pt'
FEAT_DIR      = Path('data/activations/multi_feat_L17_18_19_T5')
MLP_METRICS   = 'outputs/metrics/multi_feat_L17_18_19_T5.json'

MLP_HIDDEN    = [256, 128]
MLP_DROPOUT   = [0.5, 0.2]

SWEEP_LO      = 0.20
SWEEP_HI      = 0.80
SWEEP_STEPS   = 121   # 0.005 increments

MARK_ACCURACY = 0.41   # current accuracy-optimal threshold
MARK_SENSITIVE = 0.60  # high-sensitivity operating point


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_normalise(train_path, *split_paths):
    """Compute train mean/std, return normalised tensors for each split."""
    train_acts, train_labels = torch.load(train_path, weights_only=True)
    mean = train_acts.mean(dim=0, keepdim=True)
    std  = train_acts.std(dim=0,  keepdim=True).clamp(min=1e-8)
    out = []
    for p in split_paths:
        acts, labels = torch.load(p, weights_only=True)
        out.append(((acts - mean) / std, labels))
    return out


def run_inference(model, features, device, batch_size=512):
    """Return [n] score tensor."""
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start:start + batch_size].to(device)
            scores.append(model(batch).squeeze(1).cpu())
    return torch.cat(scores)


def sweep(scores, labels, lo, hi, steps):
    """
    Sweep threshold and return list of dicts with per-threshold metrics.
    labels: 0 = speculative, 1 = factual
    """
    thresholds = np.linspace(lo, hi, steps)
    rows = []
    n_factual     = (labels == 1).sum().item()
    n_speculative = (labels == 0).sum().item()
    n_total       = len(labels)

    for t in thresholds:
        preds = (scores >= t).long()

        # Overall accuracy
        overall_acc = (preds == labels).float().mean().item()

        # Factual accuracy  (true factual predicted as factual)
        fact_correct = ((preds == 1) & (labels == 1)).sum().item()
        fact_acc     = fact_correct / n_factual if n_factual else 0.0

        # Speculative accuracy  (true speculative predicted as speculative)
        spec_correct = ((preds == 0) & (labels == 0)).sum().item()
        spec_acc     = spec_correct / n_speculative if n_speculative else 0.0

        # FM counts
        fm1 = ((preds == 0) & (labels == 1)).sum().item()   # fact → spec
        fm2 = ((preds == 1) & (labels == 0)).sum().item()   # spec → fact

        rows.append({
            'threshold':        round(float(t), 4),
            'overall_acc':      round(overall_acc, 6),
            'factual_acc':      round(fact_acc, 6),
            'speculative_acc':  round(spec_acc, 6),
            'fm1_count':        fm1,
            'fm2_count':        fm2,
            'fm1_rate':         round(fm1 / n_factual,     6) if n_factual     else 0.0,
            'fm2_rate':         round(fm2 / n_speculative, 6) if n_speculative else 0.0,
        })

    return rows


def closest(rows, target_t):
    """Return the sweep row whose threshold is nearest to target_t."""
    return min(rows, key=lambda r: abs(r['threshold'] - target_t))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load test data for display
    test_data = load_dataset_from_file('data/splits/test.json')
    logging.info(f"Test set: {len(test_data)} examples")

    # Load and normalise features (val used only for annotation, not tuning)
    logging.info("Loading features...")
    [(val_acts, val_labels), (test_acts, test_labels)] = load_and_normalise(
        FEAT_DIR / 'train.pt',
        FEAT_DIR / 'val.pt',
        FEAT_DIR / 'test.pt',
    )
    MLP_INPUT_DIM = test_acts.shape[1]

    # Load MLP
    logging.info(f"Loading MLP from {MLP_CKPT}")
    mlp = EpistemicPredictor(
        input_dim=MLP_INPUT_DIM,
        hidden_dims=MLP_HIDDEN,
        dropout=MLP_DROPOUT,
    ).to(device)
    mlp.load_state_dict(torch.load(MLP_CKPT, weights_only=True))

    # Inference
    logging.info("Running inference on test set...")
    test_scores = run_inference(mlp, test_acts, device)
    test_labels_long = test_labels.long()

    # Sweep
    logging.info(f"Sweeping threshold {SWEEP_LO:.2f} → {SWEEP_HI:.2f} ({SWEEP_STEPS} steps)...")
    rows = sweep(test_scores, test_labels_long, SWEEP_LO, SWEEP_HI, SWEEP_STEPS)

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 75)
    print(f"  {'Threshold':>9}  {'Overall':>7}  {'Factual':>7}  {'Spec':>7}  {'FM1':>5}  {'FM2':>5}")
    print(f"  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*5}")

    # Print every 0.05 increment for readability
    display_thresholds = np.arange(SWEEP_LO, SWEEP_HI + 0.001, 0.05)
    for t in display_thresholds:
        r = closest(rows, t)
        marker = ''
        if abs(r['threshold'] - MARK_ACCURACY)  < 0.003: marker = '  ← accuracy-optimal'
        if abs(r['threshold'] - MARK_SENSITIVE) < 0.003: marker = '  ← high-sensitivity'
        print(
            f"  {r['threshold']:>9.2f}  "
            f"{r['overall_acc']:>7.2%}  "
            f"{r['factual_acc']:>7.2%}  "
            f"{r['speculative_acc']:>7.2%}  "
            f"{r['fm1_count']:>5d}  "
            f"{r['fm2_count']:>5d}"
            f"{marker}"
        )
    print("=" * 75)

    # Detailed snapshots at key operating points
    r_opt  = closest(rows, MARK_ACCURACY)
    r_sens = closest(rows, MARK_SENSITIVE)

    print(f"\n  ACCURACY-OPTIMAL  (threshold = {r_opt['threshold']:.2f})")
    print(f"    Overall     : {r_opt['overall_acc']:.2%}")
    print(f"    Factual     : {r_opt['factual_acc']:.2%}")
    print(f"    Speculative : {r_opt['speculative_acc']:.2%}")
    print(f"    FM1 (fact→spec)  : {r_opt['fm1_count']}  ({r_opt['fm1_rate']:.1%} of factual)")
    print(f"    FM2 (spec→fact)  : {r_opt['fm2_count']}  ({r_opt['fm2_rate']:.1%} of speculative)")

    print(f"\n  HIGH-SENSITIVITY  (threshold = {r_sens['threshold']:.2f})")
    print(f"    Overall     : {r_sens['overall_acc']:.2%}")
    print(f"    Factual     : {r_sens['factual_acc']:.2%}")
    print(f"    Speculative : {r_sens['speculative_acc']:.2%}")
    print(f"    FM1 (fact→spec)  : {r_sens['fm1_count']}  ({r_sens['fm1_rate']:.1%} of factual)")
    print(f"    FM2 (spec→fact)  : {r_sens['fm2_count']}  ({r_sens['fm2_rate']:.1%} of speculative)")

    fm2_reduction = r_opt['fm2_count'] - r_sens['fm2_count']
    fm1_cost      = r_sens['fm1_count'] - r_opt['fm1_count']
    acc_cost      = r_opt['overall_acc'] - r_sens['overall_acc']
    print(f"\n  Trade-off (opt → sensitive):")
    print(f"    Hallucinations caught additionally : +{fm2_reduction}  "
          f"({fm2_reduction/max(r_opt['fm2_count'],1):.0%} reduction in slip-throughs)")
    print(f"    Extra false double-checks          : +{fm1_cost}  "
          f"({fm1_cost/max(r_opt['fm1_count'],1):.0%} increase in unnecessary flags)")
    print(f"    Overall accuracy cost              : {acc_cost:.2%}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    thresholds = [r['threshold'] for r in rows]
    overall    = [r['overall_acc']     for r in rows]
    factual    = [r['factual_acc']     for r in rows]
    speculative = [r['speculative_acc'] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(thresholds, overall,    color='steelblue',  lw=2.0, label='Overall accuracy')
    ax.plot(thresholds, factual,    color='seagreen',   lw=1.8, label='Factual accuracy  (1 − FM1 rate)')
    ax.plot(thresholds, speculative, color='tomato',    lw=1.8, label='Speculative accuracy  (1 − FM2 rate)')

    # Annotate operating points
    for t, label, color in [
        (MARK_ACCURACY,  f'Accuracy-optimal\n(t={MARK_ACCURACY})', 'steelblue'),
        (MARK_SENSITIVE, f'High-sensitivity\n(t={MARK_SENSITIVE})', 'tomato'),
    ]:
        r = closest(rows, t)
        ax.axvline(t, color=color, lw=1.2, ls='--', alpha=0.7)
        ax.annotate(
            label,
            xy=(t, r['overall_acc']),
            xytext=(t + 0.02, r['overall_acc'] - 0.04),
            fontsize=9,
            color=color,
            arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
        )

    ax.set_xlabel('Decision threshold', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(
        'MLP Threshold Sensitivity\n'
        'Layers [17,18,19] · 5-token pool · test set (n=1,869)',
        fontsize=13
    )
    ax.set_xlim(SWEEP_LO, SWEEP_HI)
    ax.set_ylim(0.70, 1.01)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(alpha=0.25)

    Path('outputs/visualizations').mkdir(parents=True, exist_ok=True)
    plot_path = 'outputs/visualizations/threshold_analysis.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved plot to {plot_path}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        'model':       'multi_feat_L17_18_19_T5',
        'n_test':      len(test_data),
        'n_factual':   int((test_labels_long == 1).sum()),
        'n_speculative': int((test_labels_long == 0).sum()),
        'operating_points': {
            'accuracy_optimal': r_opt,
            'high_sensitivity': r_sens,
        },
        'sweep': rows,
    }
    Path('outputs/metrics').mkdir(parents=True, exist_ok=True)
    json_path = 'outputs/metrics/threshold_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    logging.info(f"Saved sweep data to {json_path}")


if __name__ == '__main__':
    main()
