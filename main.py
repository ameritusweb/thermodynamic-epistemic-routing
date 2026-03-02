#!/usr/bin/env python3
"""
Neural Pathway Routing PoC - Main Execution Script

This script orchestrates all phases of the epistemic routing training:
1. Dataset generation (or loading)
2. Predictor training on frozen model
3. LoRA fine-tuning with dual loss
4. Evaluation and visualization

Usage:
    python main.py --phase all
    python main.py --phase predictor
    python main.py --phase lora
    python main.py --phase eval
"""

import argparse
import logging
import yaml
import torch
from pathlib import Path
from dotenv import load_dotenv

# Import utilities
from src.utils.seed import set_global_seed
from src.utils.logger import ExperimentLogger
from src.utils.gpu_monitor import GPUMonitor


def setup_logging(config):
    """Set up logging for the experiment."""
    experiment_logger = ExperimentLogger(
        experiment_name=config['experiment']['name'],
        log_dir=f"{config['experiment']['output_dir']}/logs",
        use_wandb=config['training']['logging'].get('use_wandb', False)
    )
    return experiment_logger


def load_config(config_path: str = "config/base_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def phase_data_generation(config, args):
    """Phase 1: Generate oracle dataset."""
    logging.info("=" * 60)
    logging.info("PHASE 1: ORACLE DATASET GENERATION")
    logging.info("=" * 60)

    from src.data.question_generator import generate_oracle_dataset

    # Check if dataset already exists
    dataset_path = Path("data/processed/oracle_dataset.json")
    if dataset_path.exists() and not args.force:
        logging.info(f"Dataset already exists at {dataset_path}")
        logging.info("Use --force to regenerate")
        return

    # Generate dataset
    logging.info(f"Generating {config['data']['num_contexts']} question pairs...")
    generate_oracle_dataset(
        num_contexts=config['data']['num_contexts'],
        output_path=str(dataset_path),
        api_provider=config['api']['provider'],
        api_model=config['api']['model'],
        batch_size=config['data']['batch_size_generation'],
        batch_id=args.batch_id,
        batch_file=args.batch_file
    )

    logging.info(f"✓ Dataset generated: {dataset_path}")


def phase_predictor_training(config, args):
    """Phase 2: Train predictor on frozen model."""
    logging.info("=" * 60)
    logging.info("PHASE 2: PREDICTOR TRAINING (FROZEN MODEL)")
    logging.info("=" * 60)

    from src.training.phase1_predictor import train_predictor

    # Train predictor
    predictor, metrics = train_predictor(config)

    # Check if minimum accuracy achieved
    if metrics['best_accuracy'] < config['predictor']['training']['min_accuracy']:
        logging.warning(
            f"Predictor accuracy ({metrics['best_accuracy']:.3f}) below target "
            f"({config['predictor']['training']['min_accuracy']:.3f})"
        )
        if not args.force:
            raise ValueError("Predictor failed to reach minimum accuracy. Use --force to continue anyway.")

    logging.info(f"✓ Predictor trained successfully")
    logging.info(f"  Best accuracy: {metrics['best_accuracy']:.3f}")


def phase_lora_training(config, args):
    """Phase 3: LoRA fine-tuning with dual loss."""
    logging.info("=" * 60)
    logging.info("PHASE 3: LORA FINE-TUNING (DUAL LOSS)")
    logging.info("=" * 60)

    from src.training.phase2_lora import train_lora

    # Train with LoRA
    model, metrics = train_lora(config)

    logging.info(f"✓ LoRA training completed")
    logging.info(f"  Final generation loss: {metrics.get('final_generation_loss', 'N/A')}")
    logging.info(f"  Final routing loss: {metrics.get('final_routing_loss', 'N/A')}")


def phase_evaluation(config, args):
    """Phase 4: Evaluation and topology visualization."""
    logging.info("=" * 60)
    logging.info("PHASE 4: EVALUATION & TOPOLOGY ANALYSIS")
    logging.info("=" * 60)

    from src.evaluation.topology_visualizer import analyze_topology
    from src.evaluation.metrics_calculator import compute_separation_metrics

    # Analyze topology
    results = analyze_topology(config)

    logging.info(f"✓ Evaluation completed")
    for key, val in results.items():
        if isinstance(val, float):
            logging.info(f"  {key}: {val:.3f}")
        elif isinstance(val, dict):
            logging.info(f"  {key}: {val}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Neural Pathway Routing PoC")

    parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "data", "predictor", "lora", "eval"],
        default="all",
        help="Which phase to run"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/base_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution even if outputs exist"
    )

    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data generation phase"
    )

    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Anthropic batch ID to retrieve results from (used with --phase data)"
    )

    parser.add_argument(
        "--batch-file",
        type=str,
        default=None,
        help="Path to a locally saved batch results JSONL file (skips API retrieval)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_global_seed(config['experiment']['seed'])

    # Set up logging
    experiment_logger = setup_logging(config)

    # Log GPU info
    gpu_monitor = GPUMonitor()
    device_info = gpu_monitor.get_device_info()
    logging.info("GPU Information:")
    for key, value in device_info.items():
        logging.info(f"  {key}: {value}")

    # Execute requested phases
    try:
        if args.phase in ["all", "data"] and not args.skip_data:
            phase_data_generation(config, args)

        if args.phase in ["all", "predictor"]:
            phase_predictor_training(config, args)

        if args.phase in ["all", "lora"]:
            phase_lora_training(config, args)

        if args.phase in ["all", "eval"]:
            phase_evaluation(config, args)

        logging.info("=" * 60)
        logging.info("✓ ALL PHASES COMPLETED SUCCESSFULLY")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"Error during execution: {e}", exc_info=True)
        raise

    finally:
        experiment_logger.finish()


if __name__ == "__main__":
    main()
