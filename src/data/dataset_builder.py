"""Dataset builder for loading and preprocessing data."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import load_dataset
import random


def load_squad_contexts(num_contexts: int = 10000, min_length: int = 200, max_length: int = 1000) -> List[str]:
    """
    Load contexts from SQuAD dataset.

    Args:
        num_contexts: Number of contexts to extract
        min_length: Minimum context length
        max_length: Maximum context length

    Returns:
        List of context strings
    """
    logging.info(f"Loading SQuAD dataset...")

    # Load SQuAD v2
    dataset = load_dataset("squad_v2", split="train")

    # Extract unique contexts
    contexts = []
    seen = set()

    for example in dataset:
        context = example['context'].strip()

        # Filter by length
        if not (min_length <= len(context) <= max_length):
            continue

        # Avoid duplicates
        if context in seen:
            continue

        seen.add(context)
        contexts.append(context)

        if len(contexts) >= num_contexts:
            break

    logging.info(f"✓ Loaded {len(contexts)} unique contexts from SQuAD")

    return contexts


def create_train_val_test_splits(
    dataset: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train/val/test sets.

    Args:
        dataset: Full dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)
    shuffled = random.sample(dataset, len(dataset))

    n_train = int(len(shuffled) * train_ratio)
    n_val = int(len(shuffled) * val_ratio)

    train_data = shuffled[:n_train]
    val_data = shuffled[n_train:n_train + n_val]
    test_data = shuffled[n_train + n_val:]

    logging.info(f"✓ Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    return train_data, val_data, test_data


def save_dataset(dataset: List[Dict], output_path: str):
    """Save dataset to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logging.info(f"✓ Dataset saved: {output_path}")


def load_dataset_from_file(file_path: str) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    logging.info(f"✓ Loaded {len(dataset)} examples from {file_path}")

    return dataset


if __name__ == "__main__":
    # Test dataset builder
    logging.basicConfig(level=logging.INFO)

    contexts = load_squad_contexts(num_contexts=100)
    print(f"\nSample context:\n{contexts[0][:200]}...")
    print(f"\n✓ Dataset builder test passed")
