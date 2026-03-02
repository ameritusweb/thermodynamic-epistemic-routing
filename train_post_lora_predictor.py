"""
Train a fresh predictor on post-LoRA activations and compare to baseline.

This is the fair accuracy test: freeze the LoRA model, extract its activations,
train a new predictor from scratch on those activations — same setup as Phase 1
but on the evolved feature space. The resulting accuracy shows what the LoRA
training actually bought in terms of discriminability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.models.predictor import EpistemicPredictor
from src.training.phase1_predictor import extract_all_activations
from src.data.dataset_builder import load_dataset_from_file


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = yaml.safe_load(open('config/base_config.yaml'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load frozen LoRA model for activation extraction
    logging.info("Loading fine-tuned LoRA model (frozen)...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    base_model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model = PeftModel.from_pretrained(base_model, 'outputs/checkpoints/lora_final')
    model.eval()

    # Load datasets
    train_data = load_dataset_from_file('data/splits/train.json')
    val_data = load_dataset_from_file('data/splits/val.json')
    test_data = load_dataset_from_file('data/splits/test.json')

    # Extract activations (cached separately from baseline)
    cache_path = Path('data/activations/lora')
    cache_path.mkdir(parents=True, exist_ok=True)

    train_cache = cache_path / 'train_activations.pt'
    val_cache = cache_path / 'val_activations.pt'
    test_cache = cache_path / 'test_activations.pt'

    if train_cache.exists() and val_cache.exists() and test_cache.exists():
        logging.info("Loading cached post-LoRA activations...")
        train_activations = torch.load(train_cache, weights_only=True)
        val_activations = torch.load(val_cache, weights_only=True)
        test_activations = torch.load(test_cache, weights_only=True)
    else:
        logging.info("Extracting post-LoRA activations (train/val/test)...")
        train_activations = extract_all_activations(model, tokenizer, train_data, device=device)
        val_activations = extract_all_activations(model, tokenizer, val_data, device=device)
        test_activations = extract_all_activations(model, tokenizer, test_data, device=device)
        torch.save(train_activations, train_cache)
        torch.save(val_activations, val_cache)
        torch.save(test_activations, test_cache)
        logging.info(f"Cached to {cache_path}")

    # Free model memory before training
    del model, base_model
    torch.cuda.empty_cache()

    # Build dataloaders
    batch_size = config['predictor']['training']['batch_size']
    train_loader = DataLoader(TensorDataset(*train_activations), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_activations), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(*test_activations), batch_size=batch_size)

    # Fresh predictor
    input_dim = train_activations[0].shape[1]
    predictor = EpistemicPredictor(
        input_dim=input_dim,
        hidden_dims=config['predictor']['hidden_dims'],
        dropout=config['predictor']['dropout']
    ).to(device)
    logging.info(f"Predictor parameters: {predictor.count_parameters():,}")

    # Upweight speculative (label=0) to close the factual/speculative accuracy gap
    speculative_weight = 1.5
    bce_unreduced = nn.BCELoss(reduction='none')
    optimizer = optim.AdamW(
        predictor.parameters(),
        lr=config['predictor']['training']['learning_rate'],
        weight_decay=config['predictor']['training']['weight_decay']
    )

    def weighted_loss(preds, labels):
        weights = torch.where(labels == 0,
                              torch.full_like(labels, speculative_weight),
                              torch.ones_like(labels))
        return (bce_unreduced(preds, labels) * weights).mean()

    # Training loop
    best_accuracy = 0.0
    patience_counter = 0
    patience = config['predictor']['training']['early_stopping_patience']
    checkpoint = 'outputs/checkpoints/predictor_post_lora_best.pt'

    for epoch in range(config['predictor']['training']['epochs']):
        predictor.train()
        train_loss = 0.0
        for batch_acts, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_acts = batch_acts.to(device).float()
            batch_labels = batch_labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = weighted_loss(predictor(batch_acts), batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        predictor.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_acts, batch_labels in val_loader:
                batch_acts = batch_acts.to(device).float()
                batch_labels = batch_labels.to(device).unsqueeze(1)
                preds = predictor(batch_acts)
                val_loss += weighted_loss(preds, batch_labels).item()
                correct += ((preds >= 0.5).long() == batch_labels.long()).sum().item()
                total += batch_labels.size(0)
        val_loss /= len(val_loader)
        val_accuracy = correct / total

        logging.info(f"Epoch {epoch+1}/{config['predictor']['training']['epochs']} - "
                     f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(predictor.state_dict(), checkpoint)
            logging.info(f"  New best: {best_accuracy:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    # Tune decision threshold on val set
    predictor.load_state_dict(torch.load(checkpoint, weights_only=True))
    predictor.eval()

    all_val_scores, all_val_labels = [], []
    with torch.no_grad():
        for batch_acts, batch_labels in val_loader:
            scores = predictor(batch_acts.to(device).float()).squeeze().cpu()
            all_val_scores.append(scores)
            all_val_labels.append(batch_labels)
    all_val_scores = torch.cat(all_val_scores)
    all_val_labels = torch.cat(all_val_labels)

    best_threshold, best_val_acc = 0.5, 0.0
    for t in [i / 100 for i in range(20, 80)]:
        acc = ((all_val_scores >= t).long() == all_val_labels.long()).float().mean().item()
        if acc > best_val_acc:
            best_val_acc, best_threshold = acc, t
    logging.info(f"Optimal threshold: {best_threshold:.2f} (val acc: {best_val_acc:.4f})")

    # Final test evaluation
    correct, total = 0, 0
    factual_correct, factual_total = 0, 0
    speculative_correct, speculative_total = 0, 0

    with torch.no_grad():
        for batch_acts, batch_labels in test_loader:
            batch_acts = batch_acts.to(device).float()
            batch_labels_dev = batch_labels.to(device).unsqueeze(1)
            preds = (predictor(batch_acts) >= best_threshold).long()
            correct += (preds == batch_labels_dev.long()).sum().item()
            total += batch_labels.size(0)
            factual_mask = batch_labels == 1
            speculative_mask = batch_labels == 0
            factual_correct += (preds.cpu()[factual_mask] == batch_labels[factual_mask].unsqueeze(1)).sum().item()
            factual_total += factual_mask.sum().item()
            speculative_correct += (preds.cpu()[speculative_mask] == batch_labels[speculative_mask].unsqueeze(1)).sum().item()
            speculative_total += speculative_mask.sum().item()

    test_accuracy = correct / total
    factual_acc = factual_correct / factual_total
    speculative_acc = speculative_correct / speculative_total

    print("\n" + "=" * 50)
    print("POST-LoRA PREDICTOR RESULTS")
    print("=" * 50)
    print(f"  Threshold:            {best_threshold:.2f} (tuned on val set)")
    print(f"  Test accuracy:        {test_accuracy:.4f}")
    print(f"  Factual accuracy:     {factual_acc:.4f}")
    print(f"  Speculative accuracy: {speculative_acc:.4f}")
    print(f"\n  Baseline (Phase 1):   0.8590")
    print(f"  Improvement:          {test_accuracy - 0.8590:+.4f}")
    print("=" * 50)

    results = {
        'test_accuracy': test_accuracy,
        'factual_accuracy': factual_acc,
        'speculative_accuracy': speculative_acc,
        'threshold': best_threshold,
        'baseline_accuracy': 0.8590,
        'improvement': test_accuracy - 0.8590
    }
    Path('outputs/metrics').mkdir(parents=True, exist_ok=True)
    with open('outputs/metrics/predictor_post_lora_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    logging.info("Saved: outputs/metrics/predictor_post_lora_metrics.json")


if __name__ == '__main__':
    main()
