"""
Train a predictor on multi-layer, multi-token features from the LoRA model.

Reads layer_sweep_frozen.json to automatically pick the top 3 layers by
linear probe accuracy, then extracts mean-pooled last-N-token activations
from those layers simultaneously. The concatenated feature vector is
3× richer than the single-layer baseline.

Architecture change vs Phase 3:
    Input dim : 3 × 1536 = 4608  (vs 1536)
    Hidden    : [1024, 512, 256]  (vs [512, 256, 128])

Usage:
    python train_multi_feature_predictor.py               # top 3 from sweep, n_tokens=5
    python train_multi_feature_predictor.py --layers 14 20 26 --n-tokens 3
    python train_multi_feature_predictor.py --top-k 2     # use top 2 sweep layers
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.models.predictor import EpistemicPredictor
from src.models.multi_feature_extractor import MultiFeatureExtractor
from src.data.dataset_builder import load_dataset_from_file

LORA_PATH = 'outputs/checkpoints/lora_final'
SWEEP_JSON = 'outputs/metrics/layer_sweep_frozen.json'
BASELINE_ACCURACY = 0.9045   # Phase 3 Run 2 result


def load_top_layers(top_k: int) -> list:
    """Read layer sweep results and return the top-k layer indices."""
    if not Path(SWEEP_JSON).exists():
        raise FileNotFoundError(
            f"Layer sweep results not found at {SWEEP_JSON}. "
            "Run `python layer_sweep.py` first."
        )
    with open(SWEEP_JSON) as f:
        data = json.load(f)
    top = sorted(data['results'], key=lambda r: r['val_accuracy'], reverse=True)[:top_k]
    layers = sorted([r['layer'] for r in top])   # ascending order
    logging.info(f"Top-{top_k} layers from sweep: {layers}")
    for r in top:
        logging.info(f"  Layer {r['layer']:2d}: acc={r['val_accuracy']:.4f}  sil={r['silhouette']:+.4f}")
    return layers


def extract_split(model, tokenizer, dataset, layer_indices, n_tokens, device, cache_path):
    """Extract multi-layer features for a dataset split, with caching."""
    if cache_path.exists():
        logging.info(f"Loading cached features from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    texts = [
        f"Context: {ex['context']}\n\nQuestion: {ex['question']}\n\nAnswer:"
        for ex in dataset
    ]
    labels = torch.tensor([ex['epistemic_label'] for ex in dataset], dtype=torch.float32)

    extractor = MultiFeatureExtractor(model, layer_indices=layer_indices, n_tokens=n_tokens)
    features = extractor.extract_from_texts(texts, tokenizer, device=device)   # [n, feat_dim]

    result = (features, labels)
    torch.save(result, cache_path)
    logging.info(f"Cached to {cache_path}")
    return result


def weighted_bce_loss(preds, labels, speculative_weight=1.5):
    """BCE with higher weight for speculative (label=0) examples."""
    bce = nn.BCELoss(reduction='none')
    weights = torch.where(
        labels == 0,
        torch.full_like(labels, speculative_weight),
        torch.ones_like(labels)
    )
    return (bce(preds, labels) * weights).mean()


def main():
    parser = argparse.ArgumentParser(description="Multi-layer/multi-token predictor training")
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help="Layer indices to use (overrides sweep; e.g. --layers 14 20 26)")
    parser.add_argument('--top-k', type=int, default=3,
                        help="How many top sweep layers to use (default: 3)")
    parser.add_argument('--n-tokens', type=int, default=5,
                        help="Number of trailing tokens to mean-pool per layer (default: 5)")
    parser.add_argument('--frozen', action='store_true',
                        help="Use frozen base model instead of LoRA checkpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = yaml.safe_load(open('config/base_config.yaml'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Resolve which layers to use
    layer_indices = args.layers if args.layers else load_top_layers(args.top_k)
    n_tokens = args.n_tokens
    logging.info(f"Feature config: layers={layer_indices}, n_tokens={n_tokens}")

    feat_dim = len(layer_indices) * config['model']['hidden_dim']

    # Cache key encodes the feature configuration so different runs don't collide
    layer_tag = '_'.join(str(l) for l in layer_indices)
    model_tag = 'frozen' if args.frozen else 'lora'
    cache_dir = Path(f'data/activations/multi_feat_L{layer_tag}_T{n_tokens}_{model_tag}')
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_cache = cache_dir / 'train.pt'
    val_cache   = cache_dir / 'val.pt'
    test_cache  = cache_dir / 'test.pt'

    need_extraction = not (train_cache.exists() and val_cache.exists() and test_cache.exists())

    if need_extraction:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        base = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        if args.frozen:
            logging.info("Using frozen base model for feature extraction.")
            model = base
        else:
            logging.info(f"Loading LoRA weights from {LORA_PATH}...")
            model = PeftModel.from_pretrained(base, LORA_PATH)
        model.eval()
    else:
        logging.info("All feature caches found — skipping model load.")
        model = tokenizer = None

    train_data = load_dataset_from_file('data/splits/train.json')
    val_data   = load_dataset_from_file('data/splits/val.json')
    test_data  = load_dataset_from_file('data/splits/test.json')

    train_feats = extract_split(model, tokenizer, train_data, layer_indices, n_tokens, device, train_cache)
    val_feats   = extract_split(model, tokenizer, val_data,   layer_indices, n_tokens, device, val_cache)
    test_feats  = extract_split(model, tokenizer, test_data,  layer_indices, n_tokens, device, test_cache)

    if need_extraction:
        del model
        if not args.frozen:
            del base
        torch.cuda.empty_cache()

    # Compute normalisation statistics on train set
    train_acts, train_labels = train_feats
    val_acts,   val_labels   = val_feats
    test_acts,  test_labels  = test_feats

    mean = train_acts.mean(dim=0)
    std  = train_acts.std(dim=0).clamp(min=1e-8)
    train_acts = (train_acts - mean) / std
    val_acts   = (val_acts   - mean) / std
    test_acts  = (test_acts  - mean) / std

    # Dataloaders
    batch_size = config['predictor']['training']['batch_size']
    train_loader = DataLoader(TensorDataset(train_acts, train_labels), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_acts,   val_labels),   batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(test_acts,  test_labels),  batch_size=batch_size)

    # Smaller predictor: with 4608-dim features the representation is already rich.
    # The large [1024,512,256] network (5.4M params / 16k samples) overfits severely.
    # [256, 128] with strong dropout keeps ~1.2M params → ~76 params/sample.
    hidden_dims = [256, 128]
    dropout     = [0.5, 0.2]
    predictor = EpistemicPredictor(
        input_dim=feat_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)
    logging.info(f"Predictor: input_dim={feat_dim}, params={predictor.count_parameters():,}")

    optimizer = optim.AdamW(
        predictor.parameters(),
        lr=config['predictor']['training']['learning_rate'],
        weight_decay=0.05   # stronger regularisation vs default 0.01
    )

    # Training loop
    patience = config['predictor']['training']['early_stopping_patience']
    ckpt_path = f'outputs/checkpoints/predictor_multi_feat_L{layer_tag}_T{n_tokens}_{model_tag}_best.pt'
    Path('outputs/checkpoints').mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(config['predictor']['training']['epochs']):
        predictor.train()
        train_loss = 0.0
        for batch_acts, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_acts   = batch_acts.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = weighted_bce_loss(predictor(batch_acts), batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        predictor.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_acts, batch_labels in val_loader:
                batch_acts   = batch_acts.to(device)
                batch_labels = batch_labels.to(device).unsqueeze(1)
                preds = predictor(batch_acts)
                val_loss += weighted_bce_loss(preds, batch_labels).item()
                correct  += ((preds >= 0.5).long() == batch_labels.long()).sum().item()
                total    += batch_labels.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        logging.info(
            f"Epoch {epoch+1}/{config['predictor']['training']['epochs']} — "
            f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(predictor.state_dict(), ckpt_path)
            logging.info(f"  New best: {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    # Tune decision threshold on val set
    predictor.load_state_dict(torch.load(ckpt_path, weights_only=True))
    predictor.eval()

    all_scores, all_labels_val = [], []
    with torch.no_grad():
        for batch_acts, batch_labels in val_loader:
            scores = predictor(batch_acts.to(device)).squeeze().cpu()
            all_scores.append(scores)
            all_labels_val.append(batch_labels)
    all_scores = torch.cat(all_scores)
    all_labels_val = torch.cat(all_labels_val)

    best_threshold, best_thr_acc = 0.5, 0.0
    for t in [i / 100 for i in range(20, 80)]:
        acc = ((all_scores >= t).long() == all_labels_val.long()).float().mean().item()
        if acc > best_thr_acc:
            best_thr_acc, best_threshold = acc, t
    logging.info(f"Optimal threshold: {best_threshold:.2f} (val acc: {best_thr_acc:.4f})")

    # Final test evaluation
    correct = total = 0
    factual_correct = factual_total = 0
    speculative_correct = speculative_total = 0

    with torch.no_grad():
        for batch_acts, batch_labels in test_loader:
            batch_acts   = batch_acts.to(device)
            batch_labels_dev = batch_labels.to(device).unsqueeze(1)
            preds = (predictor(batch_acts) >= best_threshold).long()
            correct += (preds == batch_labels_dev.long()).sum().item()
            total   += batch_labels.size(0)

            factual_mask     = batch_labels == 1
            speculative_mask = batch_labels == 0
            factual_correct     += (preds.cpu()[factual_mask]     == batch_labels[factual_mask].unsqueeze(1)).sum().item()
            factual_total       += factual_mask.sum().item()
            speculative_correct += (preds.cpu()[speculative_mask] == batch_labels[speculative_mask].unsqueeze(1)).sum().item()
            speculative_total   += speculative_mask.sum().item()

    test_acc        = correct / total
    factual_acc     = factual_correct / factual_total
    speculative_acc = speculative_correct / speculative_total

    print("\n" + "=" * 55)
    print("MULTI-FEATURE PREDICTOR RESULTS")
    print("=" * 55)
    print(f"  Layers used   : {layer_indices}")
    print(f"  Tokens pooled : {n_tokens}")
    print(f"  Feature dim   : {feat_dim}")
    print(f"  Threshold     : {best_threshold:.2f} (tuned on val)")
    print(f"")
    print(f"  Test accuracy        : {test_acc:.4f}")
    print(f"  Factual accuracy     : {factual_acc:.4f}")
    print(f"  Speculative accuracy : {speculative_acc:.4f}")
    print(f"")
    print(f"  Baseline (Phase 3 Run 2) : {BASELINE_ACCURACY:.4f}")
    print(f"  Improvement              : {test_acc - BASELINE_ACCURACY:+.4f}")
    print("=" * 55)

    results = {
        'layers': layer_indices,
        'n_tokens': n_tokens,
        'feature_dim': feat_dim,
        'threshold': best_threshold,
        'test_accuracy': test_acc,
        'factual_accuracy': factual_acc,
        'speculative_accuracy': speculative_acc,
        'baseline_accuracy': BASELINE_ACCURACY,
        'improvement': test_acc - BASELINE_ACCURACY,
    }
    Path('outputs/metrics').mkdir(parents=True, exist_ok=True)
    out_path = f'outputs/metrics/multi_feat_L{layer_tag}_T{n_tokens}_{model_tag}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
