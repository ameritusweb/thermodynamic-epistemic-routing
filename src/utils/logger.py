"""Logging utilities for experiment tracking."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ExperimentLogger:
    """Structured logging for experiments."""

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./outputs/logs",
        use_wandb: bool = False
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up file and console logging
        self.setup_logging()

        # Initialize wandb if requested
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project="epistemic-routing", name=experiment_name)
                self.wandb = wandb
            except ImportError:
                logging.warning("wandb not installed, skipping W&B logging")
                self.use_wandb = False

    def setup_logging(self):
        """Configure Python logging."""
        log_file = self.log_dir / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )

        logging.info(f"Logging initialized for experiment: {self.experiment_name}")
        logging.info(f"Log file: {log_file}")

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to wandb and console."""
        # Console logging
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                 for k, v in metrics.items()])
        logging.info(f"Metrics - {metrics_str}")

        # Wandb logging
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

    def log_config(self, config: dict):
        """Log configuration."""
        logging.info("Configuration:")
        for key, value in config.items():
            logging.info(f"  {key}: {value}")

        if self.use_wandb:
            self.wandb.config.update(config)

    def finish(self):
        """Clean up logging."""
        if self.use_wandb:
            self.wandb.finish()
        logging.info(f"Experiment {self.experiment_name} finished")
