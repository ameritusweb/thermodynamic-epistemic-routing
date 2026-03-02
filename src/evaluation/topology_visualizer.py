"""Topology visualization and analysis."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA
import logging
from pathlib import Path

from ..models.activation_extractor import ActivationExtractor
from ..data.dataset_builder import load_dataset_from_file
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def extract_test_activations(config: dict, model_path: str = None):
    """
    Extract activations from test set.

    Args:
        config: Configuration dict
        model_path: Path to fine-tuned model (None for baseline)

    Returns:
        (activations, labels)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    if model_path is None:
        logging.info("Extracting from baseline (frozen) model")
        model = AutoModel.from_pretrained(
            config['model']['name'],
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        logging.info(f"Extracting from fine-tuned model: {model_path}")
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(config['model']['name'], torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_path)

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    # Load test data
    test_data = load_dataset_from_file("data/splits/test.json")

    # Extract activations
    extractor = ActivationExtractor(model, layer_index=-2, position="last")

    texts = [f"Context: {ex['context']}\n\nQuestion: {ex['question']}\n\nAnswer:" for ex in test_data]
    labels = np.array([ex['epistemic_label'] for ex in test_data])

    activations = extractor.extract_from_texts(
        texts[:config['evaluation']['topology']['n_samples']],
        tokenizer,
        device=device
    ).float().numpy()

    labels = labels[:config['evaluation']['topology']['n_samples']]

    return activations, labels


def visualize_topology(activations: np.ndarray, labels: np.ndarray, method: str = "tsne", config: dict = None):
    """
    Create 2D visualization of activation topology.

    Args:
        activations: Activation vectors [n_samples, hidden_dim]
        labels: Epistemic labels [n_samples]
        method: "tsne", "umap", or "pca"
        config: Configuration dict

    Returns:
        2D embeddings
    """
    logging.info(f"Computing {method.upper()} projection...")

    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=config['evaluation']['topology']['tsne_perplexity'],
            random_state=config['evaluation']['topology']['random_state']
        )
    elif method == "umap":
        reducer = UMAP(
            n_components=2,
            n_neighbors=config['evaluation']['topology']['umap_n_neighbors'],
            random_state=config['evaluation']['topology']['random_state']
        )
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=config['evaluation']['topology']['random_state'])
    else:
        raise ValueError(f"Unknown method: {method}")

    embeddings = reducer.fit_transform(activations)

    return embeddings


def plot_topology(embeddings: np.ndarray, labels: np.ndarray, title: str, output_path: str, config: dict):
    """
    Plot 2D topology with factual/speculative separation.

    Args:
        embeddings: 2D coordinates [n_samples, 2]
        labels: Epistemic labels [n_samples]
        title: Plot title
        output_path: Where to save plot
        config: Configuration dict
    """
    plt.style.use(config['evaluation']['visualization']['style'])
    fig, ax = plt.subplots(figsize=config['evaluation']['visualization']['figsize'])

    # Plot factual (blue) and speculative (red)
    factual_mask = labels == 1
    speculative_mask = labels == 0

    ax.scatter(
        embeddings[factual_mask, 0],
        embeddings[factual_mask, 1],
        c='blue',
        alpha=0.6,
        label='Factual',
        s=30
    )

    ax.scatter(
        embeddings[speculative_mask, 0],
        embeddings[speculative_mask, 1],
        c='red',
        alpha=0.6,
        label='Speculative',
        s=30
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=config['evaluation']['visualization']['dpi'], bbox_inches='tight')
    plt.close()

    logging.info(f"✓ Saved visualization: {output_path}")


def run_context_sensitivity_test(
    config: dict,
    n_samples: int = 200,
) -> dict:
    """
    Control test for data leakage via question phrasing.

    Tests whether the predictor responds to CONTEXT PRESENCE rather than
    question linguistic structure. For each example, runs the predictor twice:
      - With context:    "Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer:"
      - Without context: "Context: [REMOVED]\\n\\nQuestion: {question}\\n\\nAnswer:"

    If the signal is real (epistemic state):
      - Factual examples: score should be LOW with context, HIGH without (big positive delta)
      - Speculative examples: score should be HIGH regardless (near-zero delta)

    If the signal is phrasing-based (data leakage):
      - Both factual and speculative scores barely change when context is removed

    Args:
        config: Configuration dict
        n_samples: Number of test examples to evaluate (split evenly factual/speculative)

    Returns:
        Dictionary of sensitivity metrics
    """
    import json
    from ..models.predictor import EpistemicPredictor
    from ..data.dataset_builder import load_dataset_from_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Running context sensitivity control test...")

    # Load model and predictor
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    model = __import__('transformers').AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    predictor = EpistemicPredictor(
        input_dim=config['model']['hidden_dim'],
        hidden_dims=config['predictor']['hidden_dims'],
        dropout=config['predictor']['dropout']
    ).to(device)
    predictor.load_state_dict(
        torch.load("outputs/checkpoints/predictor_best.pt", weights_only=True)
    )
    predictor.eval()

    # Load test data — balanced sample
    test_data = load_dataset_from_file("data/splits/test.json")
    factual = [ex for ex in test_data if ex['epistemic_label'] == 1][:n_samples // 2]
    speculative = [ex for ex in test_data if ex['epistemic_label'] == 0][:n_samples // 2]
    examples = factual + speculative
    labels = [1] * len(factual) + [0] * len(speculative)

    from ..models.activation_extractor import ActivationExtractor
    extractor = ActivationExtractor(model, layer_index=-2, position="last")

    # Build text pairs: with context and without context
    texts_with = [
        f"Context: {ex['context']}\n\nQuestion: {ex['question']}\n\nAnswer:"
        for ex in examples
    ]
    texts_without = [
        f"Context: [REMOVED]\n\nQuestion: {ex['question']}\n\nAnswer:"
        for ex in examples
    ]

    # Extract activations for both variants
    logging.info(f"Extracting activations for {len(examples)} examples x 2 variants...")
    acts_with = extractor.extract_from_texts(texts_with, tokenizer, device=device).float()
    acts_without = extractor.extract_from_texts(texts_without, tokenizer, device=device).float()

    # Get predictor scores for both
    with torch.no_grad():
        scores_with = predictor(acts_with.to(device)).squeeze().cpu().numpy()
        scores_without = predictor(acts_without.to(device)).squeeze().cpu().numpy()

    labels = np.array(labels)
    factual_mask = labels == 1
    speculative_mask = labels == 0

    # Context sensitivity = score_without - score_with
    # Label convention: score=1 means factual, score=0 means speculative
    # Real signal: factual scores DROP a lot when context removed (large negative delta)
    #              speculative scores barely change (small delta, already low with or without)
    # Phrasing leakage: neither class changes much when context removed
    delta = scores_without - scores_with

    factual_delta = delta[factual_mask].mean()
    speculative_delta = delta[speculative_mask].mean()

    # Scores with/without context per class
    factual_score_with = scores_with[factual_mask].mean()
    factual_score_without = scores_without[factual_mask].mean()
    speculative_score_with = scores_with[speculative_mask].mean()
    speculative_score_without = scores_without[speculative_mask].mean()

    # Signal is real if:
    # - Factual scores drop significantly without context (abs(factual_delta) > 0.3)
    # - Factual is much more context-sensitive than speculative (ratio > 2x)
    context_sensitivity_ratio = abs(factual_delta) / (abs(speculative_delta) + 1e-8)
    signal_is_real = abs(factual_delta) > 0.3 and context_sensitivity_ratio > 2.0

    results = {
        'factual_score_with_context': float(factual_score_with),
        'factual_score_without_context': float(factual_score_without),
        'factual_context_sensitivity': float(factual_delta),
        'speculative_score_with_context': float(speculative_score_with),
        'speculative_score_without_context': float(speculative_score_without),
        'speculative_context_sensitivity': float(speculative_delta),
        'context_sensitivity_ratio': float(context_sensitivity_ratio),
        'verdict': 'REAL_SIGNAL' if signal_is_real else 'POSSIBLE_LEAKAGE',
    }

    logging.info("Context Sensitivity Results:")
    logging.info(f"  Factual     | with context: {factual_score_with:.3f} | without: {factual_score_without:.3f} | delta: {factual_delta:.3f}")
    logging.info(f"  Speculative | with context: {speculative_score_with:.3f} | without: {speculative_score_without:.3f} | delta: {speculative_delta:.3f}")
    logging.info(f"  Sensitivity ratio (factual/speculative): {context_sensitivity_ratio:.2f}x")
    logging.info(f"  Verdict: {results['verdict']}")

    if signal_is_real:
        logging.info("  Factual scores are strongly context-dependent; speculative scores are not — signal is epistemic.")
    else:
        logging.warning("  Scores do not change much when context removed — possible phrasing-based leakage.")
        logging.warning("  Consider regenerating dataset with matched question structures.")

    # Plot
    output_dir = Path("outputs/visualizations")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (mask, name, color) in zip(axes, [
        (factual_mask, "Factual", "blue"),
        (speculative_mask, "Speculative", "red")
    ]):
        ax.hist(scores_with[mask], bins=20, alpha=0.6, label="With context", color=color)
        ax.hist(scores_without[mask], bins=20, alpha=0.6, label="Without context", color="gray")
        ax.set_title(f"{name} — predictor score distribution")
        ax.set_xlabel("Predictor score (0=factual, 1=speculative)")
        ax.set_ylabel("Count")
        ax.legend()

    plt.suptitle(f"Context Sensitivity Test — Verdict: {results['verdict']}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "context_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: outputs/visualizations/context_sensitivity.png")

    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    with open("outputs/metrics/context_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def analyze_topology(config: dict):
    """
    Complete topology analysis pipeline.

    Args:
        config: Configuration dict

    Returns:
        Dictionary of results
    """
    from .metrics_calculator import compute_separation_metrics

    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Extract activations - BEFORE and AFTER training
    logging.info("Extracting baseline activations...")
    activations_before, labels = extract_test_activations(config, model_path=None)

    logging.info("Extracting fine-tuned activations...")
    activations_after, _ = extract_test_activations(config, model_path="outputs/checkpoints/lora_final")

    # Visualize with each method
    for method in config['evaluation']['topology']['methods']:
        # Before training
        embeddings_before = visualize_topology(activations_before, labels, method, config)
        plot_topology(
            embeddings_before,
            labels,
            f"Epistemic Topology - {method.upper()} (Before Training)",
            output_dir / f"topology_{method}_before.png",
            config
        )

        # After training
        embeddings_after = visualize_topology(activations_after, labels, method, config)
        plot_topology(
            embeddings_after,
            labels,
            f"Epistemic Topology - {method.upper()} (After Training)",
            output_dir / f"topology_{method}_after.png",
            config
        )

        # Compute metrics
        metrics_before = compute_separation_metrics(embeddings_before, labels)
        metrics_after = compute_separation_metrics(embeddings_after, labels)

        results[f'{method}_before'] = metrics_before
        results[f'{method}_after'] = metrics_after

        logging.info(f"{method.upper()} - Before: Silhouette={metrics_before['silhouette']:.3f}")
        logging.info(f"{method.upper()} - After: Silhouette={metrics_after['silhouette']:.3f}")

    # Run context sensitivity control test
    logging.info("Running context sensitivity control test...")
    sensitivity_results = run_context_sensitivity_test(config)
    results['context_sensitivity'] = sensitivity_results

    # Save results
    import json
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    with open("outputs/metrics/topology_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("✓ Topology visualizer module loaded")
