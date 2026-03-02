"""Phase 1: Train predictor on frozen model activations."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path
import json
from tqdm import tqdm

from ..models.predictor import EpistemicPredictor
from ..models.activation_extractor import ActivationExtractor
from ..utils.checkpoint_manager import CheckpointManager
from ..data.dataset_builder import load_dataset_from_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_all_activations(
    model,
    tokenizer,
    dataset: list,
    batch_size: int = 32,
    device: str = "cuda",
    layer_index: int = -2
) -> tuple:
    """
    Extract activations for all examples in dataset.

    Args:
        model: Frozen base model
        tokenizer: Tokenizer
        dataset: List of examples with 'context' and 'question' fields
        batch_size: Batch size for extraction
        device: Device to run on
        layer_index: Which transformer layer to extract from (-2 = penultimate, 18 = layer 18)

    Returns:
        (activations_tensor, labels_tensor)
    """
    logging.info(f"Extracting activations for {len(dataset)} examples (layer_index={layer_index})...")

    extractor = ActivationExtractor(model, layer_index=layer_index, position="last")

    # Prepare texts
    texts = [f"Context: {ex['context']}\n\nQuestion: {ex['question']}\n\nAnswer:" for ex in dataset]
    labels = [ex['epistemic_label'] for ex in dataset]

    # Extract activations
    activations = extractor.extract_from_texts(
        texts,
        tokenizer,
        batch_size=batch_size,
        device=device
    )

    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    logging.info(f"✓ Extracted activations: {activations.shape}")

    return activations, labels_tensor


def train_predictor(config: dict):
    """
    Train predictor on frozen model activations.

    Args:
        config: Configuration dictionary

    Returns:
        (predictor, metrics)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Training predictor on {device}")

    # Load base model (frozen)
    logging.info(f"Loading base model: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=torch.bfloat16 if config['model']['precision'] == 'bfloat16' else torch.float16,
        device_map="auto"
    )
    model.eval()

    # Routing layer — which transformer layer the predictor (and later the LoRA) will use
    routing_layer = config['training'].get('routing_layer', -2)
    # Use a layer-specific cache subdir so changing routing_layer doesn't reuse stale activations
    cache_tag = f"layer{routing_layer}"
    cache_path = Path(f"data/activations/{cache_tag}")
    cache_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Routing layer: {routing_layer} (cache: {cache_path})")

    # Load datasets
    train_data = load_dataset_from_file("data/splits/train.json")
    val_data = load_dataset_from_file("data/splits/val.json")

    train_cache = cache_path / "train_activations.pt"
    val_cache = cache_path / "val_activations.pt"

    if train_cache.exists() and val_cache.exists():
        logging.info("Loading cached activations...")
        train_activations = torch.load(train_cache, weights_only=True)
        val_activations = torch.load(val_cache, weights_only=True)
    else:
        # Extract and cache
        train_activations = extract_all_activations(model, tokenizer, train_data, device=device, layer_index=routing_layer)
        val_activations = extract_all_activations(model, tokenizer, val_data, device=device, layer_index=routing_layer)

        torch.save(train_activations, train_cache)
        torch.save(val_activations, val_cache)
        logging.info(f"✓ Cached activations to {cache_path}")

    # Create dataloaders
    train_dataset = TensorDataset(*train_activations)
    val_dataset = TensorDataset(*val_activations)

    train_loader = DataLoader(train_dataset, batch_size=config['predictor']['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['predictor']['training']['batch_size'])

    # Initialize predictor
    input_dim = train_activations[0].shape[1]
    predictor = EpistemicPredictor(
        input_dim=input_dim,
        hidden_dims=config['predictor']['hidden_dims'],
        dropout=config['predictor']['dropout']
    ).to(device)

    logging.info(f"Predictor parameters: {predictor.count_parameters():,}")

    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        predictor.parameters(),
        lr=config['predictor']['training']['learning_rate'],
        weight_decay=config['predictor']['training']['weight_decay']
    )

    # Training loop
    best_accuracy = 0.0
    patience_counter = 0
    patience = config['predictor']['training']['early_stopping_patience']

    checkpoint_manager = CheckpointManager()
    predictor_ckpt = f"outputs/checkpoints/predictor_best_{cache_tag}.pt"

    for epoch in range(config['predictor']['training']['epochs']):
        # Train
        predictor.train()
        train_loss = 0.0

        for batch_activations, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_activations = batch_activations.to(device).float()
            batch_labels = batch_labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            predictions = predictor(batch_activations)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        predictor.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_activations, batch_labels in val_loader:
                batch_activations = batch_activations.to(device).float()
                batch_labels = batch_labels.to(device).unsqueeze(1)

                predictions = predictor(batch_activations)
                loss = criterion(predictions, batch_labels)
                val_loss += loss.item()

                # Accuracy
                predicted_labels = (predictions >= 0.5).long()
                correct += (predicted_labels == batch_labels.long()).sum().item()
                total += batch_labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct / total

        logging.info(
            f"Epoch {epoch+1}/{config['predictor']['training']['epochs']} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(predictor.state_dict(), predictor_ckpt)
            logging.info(f"✓ New best accuracy: {best_accuracy:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load best model
    predictor.load_state_dict(torch.load(predictor_ckpt, weights_only=True))

    metrics = {
        'best_accuracy': best_accuracy,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss
    }

    # Save metrics
    with open("outputs/metrics/predictor_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    return predictor, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("✓ Phase 1 predictor training module loaded")
