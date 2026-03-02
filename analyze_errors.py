"""
Error analysis: what does the 9% look like, and do the MLP and CNN miss differently?

Runs the best MLP predictor (multi-feat [17,18,19]) and the best CNN predictor
(channel-dim=32, all layers) on the held-out test set side-by-side.

Reports:
  1. Error overlap matrix — how many examples each model gets right/wrong
  2. Ensemble accuracy (averaged scores, tuned on val) — upper bound if errors are complementary
  3. Confidence histogram — are errors near the decision boundary or confidently wrong?
  4. 2D PCA scatter — where do errors live in activation space?
  5. Sample texts from each error category

Usage:
    python analyze_errors.py
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.models.predictor import EpistemicPredictor
from src.models.cnn_predictor import EpistemicCNNPredictor
from src.data.dataset_builder import load_dataset_from_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ── File paths ────────────────────────────────────────────────────────────────

MLP_CKPT       = 'outputs/checkpoints/predictor_multi_feat_L17_18_19_T5_best.pt'
MLP_FEAT_DIR   = Path('data/activations/multi_feat_L17_18_19_T5')
MLP_METRICS    = 'outputs/metrics/multi_feat_L17_18_19_T5.json'
MLP_HIDDEN     = [256, 128]
MLP_DROPOUT    = [0.5, 0.2]

CNN_CKPT       = 'outputs/checkpoints/predictor_cnn_T5_C32_best.pt'
CNN_FEAT_DIR   = Path('data/activations/cnn_all_layers_T5')
CNN_METRICS    = 'outputs/metrics/cnn_predictor_T5_C32.json'
CNN_CHANNEL    = 32

N_EXAMPLES     = 10   # printed per error category


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_threshold(metrics_path: str, default: float = 0.5) -> float:
    try:
        with open(metrics_path) as f:
            return float(json.load(f).get('threshold', default))
    except Exception:
        return default


def load_norm_and_apply(train_path, *split_paths):
    """Load train features, compute mean/std, normalise all splits."""
    train_acts, train_labels = torch.load(train_path, weights_only=True)
    mean = train_acts.mean(dim=0, keepdim=True)
    std  = train_acts.std(dim=0,  keepdim=True).clamp(min=1e-8)

    results = [(train_acts - mean) / std, train_labels]
    for p in split_paths:
        acts, labels = torch.load(p, weights_only=True)
        results.append((acts - mean) / std)
        results.append(labels)
    return results


def run_predictor(model, features_2d, batch_size=256, device='cpu'):
    """Run inference; features_2d: [n, feat_dim]. Returns [n] float tensor."""
    model.eval()
    all_scores = []
    with torch.no_grad():
        for start in range(0, len(features_2d), batch_size):
            batch = features_2d[start:start + batch_size].to(device)
            scores = model(batch).squeeze(1).cpu()
            all_scores.append(scores)
    return torch.cat(all_scores)


def weighted_bce_loss(preds, labels, speculative_weight=1.5):
    bce = nn.BCELoss(reduction='none')
    weights = torch.where(labels == 0,
                          torch.full_like(labels, speculative_weight),
                          torch.ones_like(labels))
    return (bce(preds, labels.unsqueeze(1)) * weights.unsqueeze(1)).mean()


def tune_threshold(scores, labels, lo=0.20, hi=0.80, steps=60):
    best_acc, best_t = 0.0, 0.5
    for t in [lo + (hi - lo) * i / steps for i in range(steps + 1)]:
        acc = ((scores >= t).long() == labels.long()).float().mean().item()
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Load data ──────────────────────────────────────────────────────────────
    test_data = load_dataset_from_file('data/splits/test.json')
    val_data  = load_dataset_from_file('data/splits/val.json')
    logging.info(f"Test set: {len(test_data)}  Val set: {len(val_data)}")

    # ── Load and normalise MLP features ───────────────────────────────────────
    logging.info("Loading MLP features...")
    (mlp_train_acts, mlp_train_labels,
     mlp_val_acts,   mlp_val_labels,
     mlp_test_acts,  mlp_test_labels) = load_norm_and_apply(
        MLP_FEAT_DIR / 'train.pt',
        MLP_FEAT_DIR / 'val.pt',
        MLP_FEAT_DIR / 'test.pt',
    )
    # mlp_*_acts: [n, n_layers * hidden_dim]
    MLP_INPUT_DIM = mlp_test_acts.shape[1]

    # ── Load and normalise CNN features ───────────────────────────────────────
    logging.info("Loading CNN features...")
    (cnn_train_acts, cnn_train_labels,
     cnn_val_acts,   cnn_val_labels,
     cnn_test_acts,  cnn_test_labels) = load_norm_and_apply(
        CNN_FEAT_DIR / 'train.pt',
        CNN_FEAT_DIR / 'val.pt',
        CNN_FEAT_DIR / 'test.pt',
    )
    # cnn_*_acts: [n, n_layers, hidden_dim]
    CNN_N_LAYERS   = cnn_test_acts.shape[1]
    CNN_HIDDEN_DIM = cnn_test_acts.shape[2]

    # ── Load MLP predictor ────────────────────────────────────────────────────
    logging.info(f"Loading MLP from {MLP_CKPT}")
    mlp = EpistemicPredictor(
        input_dim=MLP_INPUT_DIM,
        hidden_dims=MLP_HIDDEN,
        dropout=MLP_DROPOUT,
    ).to(device)
    mlp.load_state_dict(torch.load(MLP_CKPT, weights_only=True))

    # ── Load CNN predictor ────────────────────────────────────────────────────
    logging.info(f"Loading CNN from {CNN_CKPT}")
    cnn = EpistemicCNNPredictor(
        n_layers=CNN_N_LAYERS,
        hidden_dim=CNN_HIDDEN_DIM,
        channel_dim=CNN_CHANNEL,
    ).to(device)
    cnn.load_state_dict(torch.load(CNN_CKPT, weights_only=True))

    # ── Inference on val + test ────────────────────────────────────────────────
    logging.info("Running inference...")
    mlp_val_scores  = run_predictor(mlp, mlp_val_acts,  device=device)
    mlp_test_scores = run_predictor(mlp, mlp_test_acts, device=device)

    cnn_val_scores  = run_predictor(cnn, cnn_val_acts,  device=device)
    cnn_test_scores = run_predictor(cnn, cnn_test_acts, device=device)

    # ── Per-model thresholds (loaded from metrics JSON, or tuned on val) ───────
    mlp_thr = load_threshold(MLP_METRICS, default=0.5)
    cnn_thr = load_threshold(CNN_METRICS, default=0.37)
    logging.info(f"MLP threshold: {mlp_thr:.2f}  CNN threshold: {cnn_thr:.2f}")

    # ── Per-model test accuracy ────────────────────────────────────────────────
    test_labels = mlp_test_labels.long()   # ground truth (same for both)

    mlp_preds = (mlp_test_scores >= mlp_thr).long()
    cnn_preds = (cnn_test_scores >= cnn_thr).long()

    mlp_acc = (mlp_preds == test_labels).float().mean().item()
    cnn_acc = (cnn_preds == test_labels).float().mean().item()
    logging.info(f"MLP test acc: {mlp_acc:.4f}   CNN test acc: {cnn_acc:.4f}")

    # ── Error overlap ─────────────────────────────────────────────────────────
    mlp_correct = (mlp_preds == test_labels)
    cnn_correct = (cnn_preds == test_labels)

    both_correct   = (mlp_correct & cnn_correct).sum().item()
    both_wrong     = (~mlp_correct & ~cnn_correct).sum().item()
    mlp_only_right = (mlp_correct & ~cnn_correct).sum().item()
    cnn_only_right = (~mlp_correct & cnn_correct).sum().item()
    n_test = len(test_labels)

    print("\n" + "=" * 60)
    print("ERROR OVERLAP MATRIX")
    print("=" * 60)
    print(f"  Both correct           : {both_correct:4d} / {n_test}  ({both_correct/n_test:.1%})")
    print(f"  Both wrong             : {both_wrong:4d} / {n_test}  ({both_wrong/n_test:.1%})")
    print(f"  MLP correct, CNN wrong : {mlp_only_right:4d} / {n_test}  ({mlp_only_right/n_test:.1%})")
    print(f"  CNN correct, MLP wrong : {cnn_only_right:4d} / {n_test}  ({cnn_only_right/n_test:.1%})")
    print(f"\n  Complementary errors   : {mlp_only_right + cnn_only_right} examples")
    print(f"  (Upper bound if perfectly ensembled: {(both_correct + mlp_only_right + cnn_only_right)/n_test:.1%})")

    # ── Ensemble: tune weight on val ──────────────────────────────────────────
    best_w, best_ens_val_acc = 0.5, 0.0
    for w in [i / 20 for i in range(21)]:
        ens_val = w * mlp_val_scores + (1 - w) * cnn_val_scores
        ens_thr, ens_acc = tune_threshold(ens_val, mlp_val_labels.long())
        if ens_acc > best_ens_val_acc:
            best_ens_val_acc, best_w, best_ens_thr = ens_acc, w, ens_thr

    ens_test = best_w * mlp_test_scores + (1 - best_w) * cnn_test_scores
    ens_preds = (ens_test >= best_ens_thr).long()
    ens_acc   = (ens_preds == test_labels).float().mean().item()

    factual_mask     = test_labels == 1
    speculative_mask = test_labels == 0
    ens_fact_acc = (ens_preds[factual_mask]     == test_labels[factual_mask]).float().mean().item()
    ens_spec_acc = (ens_preds[speculative_mask] == test_labels[speculative_mask]).float().mean().item()

    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS")
    print("=" * 60)
    print(f"  Weight: {best_w:.2f} × MLP + {1-best_w:.2f} × CNN  (tuned on val)")
    print(f"  Threshold: {best_ens_thr:.2f}")
    print(f"  Test accuracy        : {ens_acc:.4f}")
    print(f"  Factual accuracy     : {ens_fact_acc:.4f}")
    print(f"  Speculative accuracy : {ens_spec_acc:.4f}")
    print(f"\n  MLP alone : {mlp_acc:.4f}")
    print(f"  CNN alone : {cnn_acc:.4f}")
    print(f"  Ensemble  : {ens_acc:.4f}  ({ens_acc - max(mlp_acc, cnn_acc):+.4f} vs best single)")

    # ── Confidence analysis ────────────────────────────────────────────────────
    mlp_wrong_mask = ~mlp_correct
    cnn_wrong_mask = ~cnn_correct

    # Calibrated confidence: distance from 0.5 using own threshold
    mlp_margin = (mlp_test_scores - mlp_thr).abs()
    cnn_margin = (cnn_test_scores - cnn_thr).abs()

    print("\n" + "=" * 60)
    print("CONFIDENCE ANALYSIS  (margin = |score − threshold|)")
    print("=" * 60)
    print(f"  MLP  correct — mean margin: {mlp_margin[mlp_correct].mean():.3f}")
    print(f"  MLP  wrong   — mean margin: {mlp_margin[mlp_wrong_mask].mean():.3f}")
    print(f"  CNN  correct — mean margin: {cnn_margin[cnn_correct].mean():.3f}")
    print(f"  CNN  wrong   — mean margin: {cnn_margin[cnn_wrong_mask].mean():.3f}")

    # Fraction of errors that are high-confidence (margin > 0.2)
    mlp_conf_wrong = (mlp_wrong_mask & (mlp_margin > 0.2)).sum().item()
    cnn_conf_wrong = (cnn_wrong_mask & (cnn_margin > 0.2)).sum().item()
    print(f"\n  MLP high-confidence errors (margin > 0.2): {mlp_conf_wrong} / {mlp_wrong_mask.sum().item()}")
    print(f"  CNN high-confidence errors (margin > 0.2): {cnn_conf_wrong} / {cnn_wrong_mask.sum().item()}")

    # ── Sample texts ──────────────────────────────────────────────────────────
    def print_examples(indices, title, n=N_EXAMPLES):
        print(f"\n{'─'*60}")
        print(f"  {title}  (showing {min(n, len(indices))} of {len(indices)})")
        print(f"{'─'*60}")
        # Sort by MLP margin ascending (most boundary first)
        sorted_idx = sorted(indices, key=lambda i: mlp_margin[i].item())[:n]
        for rank, i in enumerate(sorted_idx, 1):
            ex = test_data[i]
            ctx = ex['context'][:180].replace('\n', ' ')
            print(f"\n  [{rank}] true={'factual' if ex['epistemic_label'] else 'speculative'}"
                  f"  mlp={mlp_test_scores[i]:.2f}  cnn={cnn_test_scores[i]:.2f}")
            print(f"       Q: {ex['question']}")
            print(f"       A: {ex['answer']}")
            print(f"       Context: {ctx}...")

    both_wrong_idx     = torch.where(~mlp_correct & ~cnn_correct)[0].tolist()
    mlp_only_right_idx = torch.where(mlp_correct & ~cnn_correct)[0].tolist()
    cnn_only_right_idx = torch.where(~mlp_correct & cnn_correct)[0].tolist()

    # Further split both_wrong by error type
    both_wrong_fp = [i for i in both_wrong_idx if test_labels[i] == 0]   # spec → factual
    both_wrong_fn = [i for i in both_wrong_idx if test_labels[i] == 1]   # fact → speculative

    print_examples(both_wrong_fn, "BOTH WRONG: factual predicted as speculative")
    print_examples(both_wrong_fp, "BOTH WRONG: speculative predicted as factual")
    print_examples(mlp_only_right_idx, "MLP right, CNN wrong  (CNN-unique errors)")
    print_examples(cnn_only_right_idx, "CNN right, MLP wrong  (MLP-unique errors)")

    # ── 2D PCA scatter ────────────────────────────────────────────────────────
    logging.info("Computing PCA for scatter plot...")
    # Use MLP features (4608 dims → 2D via PCA)
    feats_np = mlp_test_acts.numpy()

    pca = PCA(n_components=2, random_state=42)
    feats_2d = pca.fit_transform(feats_np)
    var_explained = pca.explained_variance_ratio_.sum()

    # Build colour categories
    # MLP and CNN label arrays
    mlp_p = mlp_preds.numpy()
    cnn_p = cnn_preds.numpy()
    true  = test_labels.numpy()

    categories = np.full(len(true), -1, dtype=int)
    categories[(true == 1) & (mlp_p == 1) & (cnn_p == 1)] = 0   # both correct factual
    categories[(true == 0) & (mlp_p == 0) & (cnn_p == 0)] = 1   # both correct speculative
    categories[(true == 1) & (mlp_p == 0) & (cnn_p == 0)] = 2   # both wrong: fact→spec
    categories[(true == 0) & (mlp_p == 1) & (cnn_p == 1)] = 3   # both wrong: spec→fact
    categories[(true == 1) & (mlp_p == 1) & (cnn_p == 0)] = 4   # MLP ok, CNN wrong (fact)
    categories[(true == 1) & (mlp_p == 0) & (cnn_p == 1)] = 5   # CNN ok, MLP wrong (fact)
    categories[(true == 0) & (mlp_p == 0) & (cnn_p == 1)] = 6   # MLP ok, CNN wrong (spec)
    categories[(true == 0) & (mlp_p == 1) & (cnn_p == 0)] = 7   # CNN ok, MLP wrong (spec)

    palette = {
        0: ('steelblue',   'Correct factual (both)',        20, 0.3),
        1: ('seagreen',    'Correct speculative (both)',    20, 0.3),
        2: ('red',         'Both wrong: fact→spec',         60, 0.9),
        3: ('darkorange',  'Both wrong: spec→fact',         60, 0.9),
        4: ('royalblue',   'MLP ok, CNN wrong (fact)',      50, 0.85),
        5: ('darkblue',    'CNN ok, MLP wrong (fact)',      50, 0.85),
        6: ('limegreen',   'MLP ok, CNN wrong (spec)',      50, 0.85),
        7: ('darkgreen',   'CNN ok, MLP wrong (spec)',      50, 0.85),
    }

    fig, ax = plt.subplots(figsize=(12, 9))
    # Draw background (correct) first, errors on top
    for cat in [0, 1, 2, 3, 4, 5, 6, 7]:
        mask = categories == cat
        if mask.sum() == 0:
            continue
        color, label, size, alpha = palette[cat]
        ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1],
                   c=color, label=f"{label} (n={mask.sum()})",
                   s=size, alpha=alpha, linewidths=0)

    ax.set_title(
        f'Test set errors — MLP vs CNN\n'
        f'PCA of MLP features (4608→2D, var explained: {var_explained:.1%})',
        fontsize=13
    )
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.legend(fontsize=8, loc='upper right', markerscale=1.5)
    ax.grid(alpha=0.2)

    Path('outputs/visualizations').mkdir(parents=True, exist_ok=True)
    plot_path = 'outputs/visualizations/error_analysis.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved scatter plot to {plot_path}")

    # ── Save error details to JSON ────────────────────────────────────────────
    error_records = []
    for i in range(len(test_data)):
        ex = test_data[i]
        if mlp_p[i] != true[i] or cnn_p[i] != true[i]:
            error_records.append({
                'index':         i,
                'true_label':    int(true[i]),
                'mlp_score':     float(mlp_test_scores[i]),
                'cnn_score':     float(cnn_test_scores[i]),
                'mlp_correct':   bool(mlp_p[i] == true[i]),
                'cnn_correct':   bool(cnn_p[i] == true[i]),
                'question':      ex['question'],
                'answer':        ex['answer'],
                'context_len':   len(ex['context']),
            })

    out_json = 'outputs/metrics/error_analysis.json'
    with open(out_json, 'w') as f:
        json.dump({
            'mlp_accuracy':    mlp_acc,
            'cnn_accuracy':    cnn_acc,
            'ensemble_accuracy': ens_acc,
            'ensemble_weight_mlp': best_w,
            'both_correct':    both_correct,
            'both_wrong':      both_wrong,
            'mlp_only_right':  mlp_only_right,
            'cnn_only_right':  cnn_only_right,
            'errors':          error_records,
        }, f, indent=2)
    logging.info(f"Saved {len(error_records)} error records to {out_json}")


if __name__ == '__main__':
    main()
