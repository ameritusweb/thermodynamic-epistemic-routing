"""Metrics for quantifying epistemic separation."""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging


def compute_separation_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute quantitative separation metrics.

    Args:
        embeddings: 2D coordinates [n_samples, 2]
        labels: Binary labels [n_samples]

    Returns:
        Dictionary of metrics
    """
    # Silhouette score (higher is better, range: -1 to 1)
    silhouette = silhouette_score(embeddings, labels)

    # Davies-Bouldin index (lower is better)
    db_index = davies_bouldin_score(embeddings, labels)

    # Compute centroids
    factual_mask = labels == 1
    speculative_mask = labels == 0

    factual_centroid = embeddings[factual_mask].mean(axis=0)
    speculative_centroid = embeddings[speculative_mask].mean(axis=0)

    # Inter-cluster distance
    inter_distance = np.linalg.norm(factual_centroid - speculative_centroid)

    # Intra-cluster variance
    factual_variance = np.var(embeddings[factual_mask])
    speculative_variance = np.var(embeddings[speculative_mask])
    avg_variance = (factual_variance + speculative_variance) / 2

    # Separation ratio
    separation_ratio = inter_distance / np.sqrt(avg_variance)

    metrics = {
        'silhouette': float(silhouette),
        'davies_bouldin': float(db_index),
        'inter_cluster_distance': float(inter_distance),
        'separation_ratio': float(separation_ratio),
        'factual_variance': float(factual_variance),
        'speculative_variance': float(speculative_variance)
    }

    return metrics


def check_success_criteria(metrics: dict) -> dict:
    """
    Check if metrics meet success criteria from the paper.

    Args:
        metrics: Dictionary of computed metrics

    Returns:
        Dictionary of pass/fail for each criterion
    """
    criteria = {
        'silhouette_score': metrics['silhouette'] > 0.3,
        'separation_ratio': metrics['separation_ratio'] > 2.0
    }

    all_passed = all(criteria.values())

    logging.info("Success criteria check:")
    for criterion, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logging.info(f"  {criterion}: {status}")

    if all_passed:
        logging.info("✓ All success criteria met!")
    else:
        logging.warning("Some success criteria not met. Consider:")
        logging.warning("  - Increasing lambda_routing")
        logging.warning("  - Training longer")
        logging.warning("  - Increasing LoRA rank")

    return {'criteria': criteria, 'all_passed': all_passed}


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)

    # Simulated well-separated clusters
    n_samples = 1000
    factual = np.random.randn(n_samples // 2, 2) + np.array([3, 3])
    speculative = np.random.randn(n_samples // 2, 2) - np.array([3, 3])

    embeddings = np.vstack([factual, speculative])
    labels = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

    metrics = compute_separation_metrics(embeddings, labels)

    print("Test metrics (simulated well-separated clusters):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")

    check_success_criteria(metrics)
