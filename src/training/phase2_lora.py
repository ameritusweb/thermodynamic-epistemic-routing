"""Phase 2: LoRA fine-tuning with adversarial co-evolution and dual loss."""

import torch
import logging
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
from datasets import Dataset

from ..models.predictor import EpistemicPredictor
from ..models.lora_config import get_lora_config, apply_lora
from ..data.dataset_builder import load_dataset_from_file
from ..data.paired_sampler import PairedSampler
from .custom_trainer import EpistemicRoutingTrainer


def prepare_lora_dataset(dataset: list, tokenizer) -> Dataset:
    """
    Prepare dataset for LoRA training.

    Args:
        dataset: List of examples
        tokenizer: Tokenizer

    Returns:
        HuggingFace Dataset
    """
    def format_example(example):
        """Format as conversation."""
        text = f"Context: {example['context']}\n\nQuestion: {example['question']}\n\nAnswer: {example['answer']}"
        return {'text': text, 'epistemic_label': example['epistemic_label']}

    formatted = [format_example(ex) for ex in dataset]
    return Dataset.from_list(formatted)


def train_lora(config: dict):
    """
    Train model with LoRA using adversarial co-evolution and dual loss.

    Training alternates between:
    - Generator update every step: routing loss gradient flows back through
      activations to LoRA params. Predictor is frozen (eval mode, no optimizer call).
    - Predictor update every N steps (after warmup): activations are detached;
      predictor params updated; generator receives no gradient.

    Args:
        config: Configuration dictionary

    Returns:
        (model, metrics)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Training LoRA with adversarial co-evolution on {device}")

    adversarial_config = config['training']['adversarial']
    predictor_update_freq = adversarial_config['predictor_update_freq']
    predictor_warmup_steps = adversarial_config['predictor_warmup_steps']
    predictor_grad_clip = adversarial_config['predictor_grad_clip']

    # Load tokenizer and model
    logging.info(f"Loading model: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=torch.bfloat16 if config['model']['precision'] == 'bfloat16' else torch.float16,
        device_map="auto",
        trust_remote_code=config['model'].get('trust_remote_code', True)
    )

    # Enable gradient checkpointing before LoRA so PEFT wraps correctly
    if config['training']['optimization']['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # Required for PEFT + gradient checkpointing

    # Apply LoRA
    lora_config = get_lora_config(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules']
    )
    model = apply_lora(model, lora_config)

    # Load predictor — keep in train mode for co-evolution
    logging.info("Loading trained predictor (unfrozen for co-evolution)...")
    input_dim = config['model']['hidden_dim']
    predictor = EpistemicPredictor(
        input_dim=input_dim,
        hidden_dims=config['predictor']['hidden_dims'],
        dropout=config['predictor']['dropout']
    ).to(device)

    routing_layer = config['training'].get('routing_layer', -2)
    cache_tag = f"layer{routing_layer}"
    predictor_ckpt = f"outputs/checkpoints/predictor_best_{cache_tag}.pt"
    logging.info(f"Loading predictor warm-start from {predictor_ckpt}")
    predictor.load_state_dict(torch.load(predictor_ckpt, weights_only=True))
    predictor.train()  # Unfrozen — will co-evolve with generator

    # Separate optimizer for predictor (lower LR — warm-started from Phase 1)
    predictor_optimizer = AdamW(
        predictor.parameters(),
        lr=adversarial_config['predictor_lr_phase2'],
        weight_decay=0.01
    )

    # Load datasets
    train_data = load_dataset_from_file("data/splits/train.json")
    val_data = load_dataset_from_file("data/splits/val.json")

    train_hf_dataset = prepare_lora_dataset(train_data, tokenizer)
    val_hf_dataset = prepare_lora_dataset(val_data, tokenizer)

    # Tokenize
    def tokenize_function(examples):
        outputs = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
        outputs['epistemic_labels'] = examples['epistemic_label']
        return outputs

    train_hf_dataset = train_hf_dataset.map(tokenize_function, batched=True, remove_columns=['text', 'epistemic_label'])
    val_hf_dataset = val_hf_dataset.map(tokenize_function, batched=True, remove_columns=['text', 'epistemic_label'])

    # Training arguments (used for eval, logging, and checkpoint config)
    training_args = TrainingArguments(
        output_dir="outputs/checkpoints/lora_training",
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_ratio=config['training']['warmup_ratio'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        fp16=config['training']['optimization']['fp16'],
        bf16=config['training']['optimization']['bf16'],
        logging_steps=config['training']['logging']['logging_steps'],
        eval_strategy="steps",
        eval_steps=config['training']['logging']['eval_steps'],
        save_steps=config['training']['checkpointing']['save_steps'],
        save_total_limit=config['training']['checkpointing']['save_total_limit'],
        load_best_model_at_end=config['training']['checkpointing']['load_best_model_at_end'],
        report_to="tensorboard" if config['training']['logging']['use_tensorboard'] else "none"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    thermo_layers = adversarial_config.get('thermo_layers', None)
    lambda_thermo = adversarial_config.get('lambda_thermo', 1.0)
    thermo_margin = adversarial_config.get('thermo_margin', 0.05)

    # Custom trainer (used for compute_loss, prediction_step, and eval machinery)
    trainer = EpistemicRoutingTrainer(
        model=model,
        predictor=predictor,
        lambda_routing=config['training']['lambda_routing'],
        lambda_contrastive=adversarial_config['lambda_contrastive'],
        contrastive_margin=adversarial_config['contrastive_margin'],
        routing_layer=config['training'].get('routing_layer', -2),
        thermo_layers=thermo_layers,
        lambda_thermo=lambda_thermo,
        thermo_margin=thermo_margin,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=val_hf_dataset,
        data_collator=data_collator
    )

    # PairedSampler keeps same-context (factual, speculative) pairs contiguous in
    # each batch, enabling pairwise context-normalised turbulence computation.
    # batch_size must be even; drop_last=True guarantees complete pairs.
    paired_sampler = PairedSampler(
        train_data,
        shuffle=True,
        seed=config['experiment']['seed']
    )
    train_dataloader = DataLoader(
        train_hf_dataset,
        sampler=paired_sampler,
        batch_size=config['training']['batch_size'],
        collate_fn=data_collator,
        drop_last=True
    )

    # Set up generator optimizer and scheduler via Trainer's own machinery
    num_training_steps = len(train_dataloader) * config['training']['epochs']
    trainer.create_optimizer_and_scheduler(num_training_steps=num_training_steps)
    generator_optimizer = trainer.optimizer
    generator_scheduler = trainer.lr_scheduler

    grad_accum_steps = config['training']['gradient_accumulation_steps']
    max_grad_norm = config['training']['optimization']['max_grad_norm']
    logging_steps = config['training']['logging']['logging_steps']

    # ---- Adversarial Alternating Training Loop ----
    global_step = 0
    predictor_update_count = 0
    all_metrics = []

    # Initialise Trainer callback infrastructure so self.log() writes to TensorBoard.
    # Without this, on_train_begin is never called and the TensorBoard writer never opens.
    trainer.state.max_steps = num_training_steps
    trainer.control = trainer.callback_handler.on_train_begin(
        trainer.args, trainer.state, trainer.control
    )

    logging.info("Starting adversarial co-evolution training loop...")
    logging.info(
        f"  Predictor updates begin at step {predictor_warmup_steps}, "
        f"every {predictor_update_freq} generator steps"
    )

    for epoch in range(config['training']['epochs']):
        paired_sampler.set_epoch(epoch)
        model.train()
        predictor.train()

        epoch_gen_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = trainer._prepare_inputs(batch)

            # ---- GENERATOR UPDATE STEP ----
            # Gradient flows: routing_loss → activations → LoRA params
            # Predictor: eval mode, optimizer not called → params unchanged
            trainer._is_predictor_update_step = False

            gen_loss = trainer.compute_loss(model, batch, return_outputs=False)
            gen_loss = gen_loss / grad_accum_steps
            gen_loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                generator_optimizer.step()
                generator_scheduler.step()
                generator_optimizer.zero_grad()

            epoch_gen_loss += gen_loss.item() * grad_accum_steps  # unscale for logging
            epoch_steps += 1

            # ---- PREDICTOR UPDATE STEP ----
            # Runs every predictor_update_freq generator steps, after warmup.
            # Gradient flows: pred_loss → predictor params only (activations detached)
            # Generator receives no gradient — its optimizer is not called.
            if (
                global_step >= predictor_warmup_steps
                and global_step % predictor_update_freq == 0
            ):
                trainer._is_predictor_update_step = True

                predictor_optimizer.zero_grad()
                # Re-run forward pass with detached activations.
                # Generator params build no useful graph here; only predictor params
                # receive gradients.
                pred_loss = trainer.compute_loss(model, batch, return_outputs=False)
                pred_loss.backward()

                torch.nn.utils.clip_grad_norm_(predictor.parameters(), predictor_grad_clip)
                predictor_optimizer.step()

                epoch_pred_loss += pred_loss.item()
                predictor_update_count += 1

                trainer._is_predictor_update_step = False  # Reset flag

            global_step += 1
            trainer.state.global_step = global_step

            if global_step % logging_steps == 0:
                avg_gen = epoch_gen_loss / epoch_steps
                logging.info(
                    f"Epoch {epoch+1} | Step {global_step} | "
                    f"Avg Gen Loss: {avg_gen:.4f} | "
                    f"Predictor Updates: {predictor_update_count}"
                )

        # ---- END OF EPOCH ----
        avg_gen_loss = epoch_gen_loss / max(epoch_steps, 1)
        avg_pred_loss = epoch_pred_loss / max(predictor_update_count, 1)
        logging.info(
            f"Epoch {epoch+1} complete | "
            f"Avg Gen Loss: {avg_gen_loss:.4f} | "
            f"Avg Pred Loss: {avg_pred_loss:.4f} | "
            f"Total Predictor Updates: {predictor_update_count}"
        )

        # Evaluate using Trainer's eval machinery
        eval_results = trainer.evaluate()
        logging.info(f"Epoch {epoch+1} eval: {eval_results}")

        # Save generator and predictor checkpoints
        epoch_dir = Path(f"outputs/checkpoints/lora_epoch_{epoch+1}")
        trainer.save_model(str(epoch_dir))
        torch.save(
            predictor.state_dict(),
            epoch_dir / f"predictor_epoch_{epoch+1}.pt"
        )
        logging.info(f"✓ Checkpoints saved to {epoch_dir}")

        all_metrics.append({
            'epoch': epoch + 1,
            'avg_gen_loss': avg_gen_loss,
            'avg_pred_loss': avg_pred_loss,
            'eval': eval_results,
            'predictor_updates': predictor_update_count,
        })

    # Save final model and predictor
    final_dir = Path("outputs/checkpoints/lora_final")
    trainer.save_model(str(final_dir))
    torch.save(predictor.state_dict(), final_dir / "predictor_final.pt")
    logging.info(f"✓ Final model and predictor saved to {final_dir}")

    metrics = {
        'epochs': all_metrics,
        'total_predictor_updates': predictor_update_count,
        'final_gen_loss': all_metrics[-1]['avg_gen_loss'] if all_metrics else None,
        'final_pred_loss': all_metrics[-1]['avg_pred_loss'] if all_metrics else None,
    }

    return model, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("✓ Phase 2 adversarial LoRA training module loaded")
