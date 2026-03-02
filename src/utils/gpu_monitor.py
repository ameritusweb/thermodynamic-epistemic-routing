"""GPU monitoring utilities."""

import logging
import torch
from typing import Dict, Optional


class GPUMonitor:
    """Monitor GPU memory usage."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = torch.cuda.is_available()

    def log_memory_stats(self, prefix: str = "") -> Optional[Dict[str, float]]:
        """
        Log current GPU memory statistics.

        Args:
            prefix: Prefix for log message

        Returns:
            Dictionary of memory stats in GB
        """
        if not self.is_cuda:
            logging.info("CUDA not available")
            return None

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9

        stats = {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "peak_gb": max_allocated
        }

        msg = f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Peak: {max_allocated:.2f}GB"
        logging.info(msg)

        return stats

    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def get_device_info(self) -> Dict[str, any]:
        """Get GPU device information."""
        if not self.is_cuda:
            return {"device": "cpu"}

        return {
            "device": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
        }
