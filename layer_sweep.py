"""
Layer sweep: find which transformer layer carries the strongest epistemic signal.

Hooks ALL layers simultaneously in a single forward pass (one pass per batch),
captures last-token activations, then trains a logistic regression probe per layer.
Reports accuracy and silhouette score per layer — no retraining of LoRA needed.

Usage:
    python layer_sweep.py                          # frozen base model
    python layer_sweep.py --model-path outputs/checkpoints/lora_final   # LoRA model
"""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset_builder import load_dataset_from_file

N_SWEEP_SAMPLES = 2000   # balanced: 1000 factual + 1000 speculative
BATCH_SIZE = 32


def _find_transformer_layers(model):
    """
    Robustly locate the transformer decoder layer list in any supported architecture.

    Tries known attribute paths first, then falls back to walking named_modules()
    to find the largest ModuleList whose first element has a self-attention block.
    """
    import torch.nn as nn

    m = model
    # PEFT unwrap
    if hasattr(m, 'base_model') and m.base_model is not m and hasattr(m.base_model, 'model'):
        m = m.base_model.model

    # VLM: language_model wraps a ForCausalLM which has .model.layers
    if hasattr(m, 'language_model'):
        lm = m.language_model
        if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            return lm.model.layers        # e.g. Gemma3ForConditionalGeneration
        if hasattr(lm, 'layers'):
            return lm.layers              # language_model IS the text model

    # Standard CausalLM: model.model.layers
    if hasattr(m, 'model') and hasattr(m.model, 'layers'):
        return m.model.layers

    # Bare text model: model.layers
    if hasattr(m, 'layers'):
        return m.layers

    # GPT-2 style
    if hasattr(m, 'transformer') and hasattr(m.transformer, 'h'):
        return m.transformer.h

    # Last resort: walk all named modules for a ModuleList that looks like
    # transformer decoder layers (has self-attention as first child).
    best = None
    for name, module in m.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 10:
            first = module[0]
            if any(hasattr(first, a) for a in ('self_attn', 'attention', 'attn', 'self_attention')):
                if best is None or len(module) > len(best):
                    best = module
                    logging.debug(f"named_modules fallback candidate: '{name}' ({len(module)} layers)")
    if best is not None:
        logging.info(f"Located {len(best)} transformer layers via named_modules() fallback")
        return best

    raise ValueError(
        f"Cannot find transformer layers in {type(model).__name__}. "
        f"Top-level children: {[n for n, _ in model.named_children()]}"
    )


class LayerSweepExtractor:
    """
    Extract last-token activations from ALL layers in a single forward pass.

    Registers one hook per layer; each hook immediately reduces the full
    [batch, seq, hidden] tensor to [batch, hidden] at the last non-padding
    token, so peak extra GPU memory is negligible.
    """

    def __init__(self, model, n_layers: int):
        self.model = model
        self.n_layers = n_layers
        self.layer_activations: dict = {}
        self.hooks: list = []
        self._attention_mask = None

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            mask = self._attention_mask
            if mask is not None:
                seq_lengths = mask.sum(dim=1) - 1
                batch_idx = torch.arange(hidden.size(0), device=hidden.device)
                vec = hidden[batch_idx, seq_lengths].detach().cpu().float()
            else:
                vec = hidden[:, -1, :].detach().cpu().float()
            self.layer_activations[layer_idx] = vec
        return hook

    def _resolve_layers(self):
        return _find_transformer_layers(self.model)

    def register_hooks(self):
        layers = self._resolve_layers()
        actual = min(self.n_layers, len(layers))
        if actual != self.n_layers:
            logging.warning(f"Requested {self.n_layers} layers but model has {len(layers)}; using {actual}")
            self.n_layers = actual
        for i in range(self.n_layers):
            h = layers[i].register_forward_hook(self._make_hook(i))
            self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def extract_batch(self, input_ids, attention_mask) -> dict:
        """Run one forward pass; return {layer_idx: [batch, hidden]} for all layers."""
        self._attention_mask = attention_mask
        self.layer_activations.clear()
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        return {k: v.clone() for k, v in self.layer_activations.items()}


def extract_all_layers(model, tokenizer, examples, device) -> tuple:
    """
    Extract all-layer activations for a list of examples in a single forward pass per batch.

    Returns:
        layer_arrays: {layer_idx: np.ndarray [n_samples, hidden_dim]}
        labels: np.ndarray [n_samples]
    """
    # Auto-detect layer count using the same robust resolver
    n_layers = len(_find_transformer_layers(model))
    logging.info(f"Detected {n_layers} transformer layers")

    extractor = LayerSweepExtractor(model, n_layers)
    extractor.register_hooks()

    texts = [
        f"Context: {ex['context']}\n\nQuestion: {ex['question']}\n\nAnswer:"
        for ex in examples
    ]
    labels = np.array([ex['epistemic_label'] for ex in examples])

    layer_buffers: dict = {i: [] for i in range(n_layers)}

    try:
        for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting layers"):
            batch_texts = texts[start:start + BATCH_SIZE]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            batch_acts = extractor.extract_batch(input_ids, attention_mask)
            for i in range(n_layers):
                layer_buffers[i].append(batch_acts[i])
    finally:
        extractor.remove_hooks()

    layer_arrays = {
        i: torch.cat(layer_buffers[i], dim=0).numpy()
        for i in range(n_layers)
    }
    return layer_arrays, labels, n_layers


def probe_layer(acts: np.ndarray, labels: np.ndarray, seed: int = 42) -> tuple:
    """
    Train a logistic regression probe on layer activations.

    Returns:
        (val_accuracy, silhouette)
    """
    n = len(labels)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, X_val = acts[train_idx], acts[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    # Standardize on train statistics
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std

    probe = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, n_jobs=-1)
    probe.fit(X_train_n, y_train)
    val_acc = probe.score(X_val_n, y_val)

    # Silhouette on a capped subsample (expensive at full 400 samples)
    sil_n = min(400, len(X_val_n))
    sil_idx = rng.choice(len(X_val_n), sil_n, replace=False)
    try:
        sil = float(silhouette_score(X_val_n[sil_idx], y_val[sil_idx]))
    except Exception:
        sil = 0.0

    return float(val_acc), sil


def main():
    parser = argparse.ArgumentParser(description="Layer sweep: epistemic signal by layer depth")
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Path to LoRA checkpoint (default: frozen base model)"
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=N_SWEEP_SAMPLES,
        help=f"Total balanced samples to use (default: {N_SWEEP_SAMPLES})"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = yaml.safe_load(open('config/base_config.yaml'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_tag = 'lora' if args.model_path else 'frozen'
    cache_file = Path(f'data/activations/layer_sweep_{model_tag}.npz')
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        logging.info(f"Loading cached layer sweep activations ({cache_file})...")
        data = np.load(str(cache_file))
        labels = data['labels']
        n_layers = int(data['n_layers'])
        layer_arrays = {i: data[str(i)] for i in range(n_layers)}
    else:
        # Build balanced sweep dataset
        val_data = load_dataset_from_file('data/splits/val.json')
        test_data = load_dataset_from_file('data/splits/test.json')
        all_data = val_data + test_data
        n_per_class = args.n_samples // 2
        factual = [ex for ex in all_data if ex['epistemic_label'] == 1][:n_per_class]
        speculative = [ex for ex in all_data if ex['epistemic_label'] == 0][:n_per_class]
        examples = factual + speculative
        logging.info(
            f"Sweep dataset: {len(examples)} examples "
            f"({len(factual)} factual, {len(speculative)} speculative)"
        )

        # Load model
        logging.info(f"Loading model: {config['model']['name']} (tag={model_tag})")
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        if args.model_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, args.model_path)
            logging.info(f"Loaded LoRA weights from {args.model_path}")
        else:
            model = base_model
        model.eval()

        layer_arrays, labels, n_layers = extract_all_layers(model, tokenizer, examples, device)

        # Free GPU memory before sklearn probing
        del model, base_model
        torch.cuda.empty_cache()

        # Cache compressed
        save_dict = {'labels': labels, 'n_layers': np.array(n_layers)}
        save_dict.update({str(i): layer_arrays[i] for i in range(n_layers)})
        np.savez_compressed(str(cache_file), **save_dict)
        logging.info(f"Cached to {cache_file}")

    # Probe each layer
    logging.info(f"Training linear probes on {n_layers} layers...")
    results = []
    for layer_idx in tqdm(range(n_layers), desc="Probing"):
        val_acc, sil = probe_layer(layer_arrays[layer_idx], labels)
        results.append({'layer': layer_idx, 'val_accuracy': val_acc, 'silhouette': sil})
        logging.info(f"  Layer {layer_idx:2d}: acc={val_acc:.4f}  sil={sil:+.4f}")

    # Summarise
    best = max(results, key=lambda r: r['val_accuracy'])
    top5 = sorted(results, key=lambda r: r['val_accuracy'], reverse=True)[:5]

    print("\n" + "=" * 55)
    print(f"LAYER SWEEP RESULTS  ({model_tag.upper()} model)")
    print("=" * 55)
    print(f"Best layer : {best['layer']}  "
          f"(acc={best['val_accuracy']:.4f}, sil={best['silhouette']:+.4f})")
    print(f"\nTop 5 layers by linear probe accuracy:")
    for r in top5:
        print(f"  Layer {r['layer']:2d}: acc={r['val_accuracy']:.4f}  sil={r['silhouette']:+.4f}")
    print(f"\nBaseline (Phase 1, layer -2): 0.8590")
    print(f"Best layer vs baseline      : {best['val_accuracy'] - 0.8590:+.4f}")
    print("=" * 55)

    # Save JSON
    Path('outputs/metrics').mkdir(parents=True, exist_ok=True)
    out_path = f'outputs/metrics/layer_sweep_{model_tag}.json'
    with open(out_path, 'w') as f:
        json.dump({'model_tag': model_tag, 'results': results,
                   'best_layer': best['layer'], 'top5': top5}, f, indent=2)
    logging.info(f"Saved: {out_path}")

    # Plot
    layers = [r['layer'] for r in results]
    accuracies = [r['val_accuracy'] for r in results]
    silhouettes = [r['silhouette'] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    bars = ax1.bar(layers, accuracies, color='steelblue', alpha=0.8)
    bars[best['layer']].set_color('darkorange')
    ax1.axhline(y=0.859, color='red', linestyle='--', linewidth=1.5,
                label='Phase 1 baseline (layer −2, 85.9%)')
    ax1.axvline(x=best['layer'], color='darkorange', linestyle='--',
                linewidth=1.5, label=f"Best: layer {best['layer']} ({best['val_accuracy']:.1%})")
    ax1.set_ylabel('Linear Probe Val Accuracy', fontsize=11)
    model_short = config['model']['name'].split('/')[-1]
    ax1.set_title(
        f'Epistemic Signal by Transformer Layer — {model_short} ({model_tag})',
        fontsize=13
    )
    ax1.legend(fontsize=10)
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    ax2.bar(layers, silhouettes, color='coral', alpha=0.8)
    ax2.axvline(x=best['layer'], color='darkorange', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Layer Index', fontsize=11)
    ax2.set_ylabel('Silhouette Score', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    Path('outputs/visualizations').mkdir(parents=True, exist_ok=True)
    fig_path = f'outputs/visualizations/layer_sweep_{model_tag}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
