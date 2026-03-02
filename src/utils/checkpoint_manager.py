"""Checkpoint management for saving and loading models."""

import torch
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class CheckpointManager:
    """Manage model checkpoints and training state."""

    def __init__(self, checkpoint_dir: str = "./outputs/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        name: str = "checkpoint"
    ):
        """
        Save model checkpoint with metadata.

        Args:
            model: PyTorch model to save
            optimizer: Optimizer state (optional)
            epoch: Current epoch number
            step: Current training step
            metrics: Performance metrics
            config: Configuration dict
            name: Checkpoint name prefix
        """
        checkpoint_path = self.checkpoint_dir / f"{name}-step{step}"
        checkpoint_path.mkdir(exist_ok=True)

        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if metrics is not None:
            checkpoint['metrics'] = metrics

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path / "checkpoint.pt")

        # Save config
        if config is not None:
            with open(checkpoint_path / "config.yaml", 'w') as f:
                yaml.dump(config, f)

        # Save metrics separately
        if metrics is not None:
            with open(checkpoint_path / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)

        logging.info(f"✓ Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            model: PyTorch model to load weights into
            checkpoint_path: Path to checkpoint directory or file
            optimizer: Optimizer to load state into (optional)
            device: Device to load to

        Returns:
            Dictionary with epoch, step, metrics
        """
        checkpoint_path = Path(checkpoint_path)

        # Handle both file and directory paths
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "checkpoint.pt"
        else:
            checkpoint_file = checkpoint_path

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        checkpoint = torch.load(checkpoint_file, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logging.info(f"✓ Checkpoint loaded: {checkpoint_file}")
        logging.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Step: {checkpoint.get('step', 'N/A')}")

        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {})
        }

    def find_latest_checkpoint(self, pattern: str = "checkpoint-step*") -> Optional[Path]:
        """
        Find the most recent checkpoint matching pattern.

        Args:
            pattern: Glob pattern for checkpoint directories

        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        if not checkpoints:
            return None

        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest

    def list_checkpoints(self, pattern: str = "checkpoint-step*") -> list:
        """List all checkpoints matching pattern."""
        return sorted(self.checkpoint_dir.glob(pattern))

    def delete_old_checkpoints(self, keep_last_n: int = 5, pattern: str = "checkpoint-step*"):
        """Delete old checkpoints, keeping only the most recent N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for checkpoint in checkpoints[keep_last_n:]:
            import shutil
            shutil.rmtree(checkpoint)
            logging.info(f"Deleted old checkpoint: {checkpoint.name}")
