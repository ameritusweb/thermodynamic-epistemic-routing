"""MLP Predictor for epistemic state classification."""

import torch
import torch.nn as nn
from typing import List


class EpistemicPredictor(nn.Module):
    """
    MLP classifier for predicting epistemic state (factual vs. speculative).

    Takes activation vectors from the generator's penultimate layer and
    classifies them as factual (1) or speculative (0).
    """

    def __init__(
        self,
        input_dim: int = 1536,  # Qwen2.5-1.5B hidden size
        hidden_dims: List[int] = [512, 256, 128],
        dropout: List[float] = [0.3, 0.2, 0.0],
        use_layer_norm: bool = True
    ):
        """
        Initialize predictor network.

        Args:
            input_dim: Dimension of input activation vectors
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rates for each hidden layer
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build network layers
        layers = []
        prev_dim = input_dim

        for i, (hidden_dim, drop_rate) in enumerate(zip(hidden_dims, dropout)):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Layer normalization (optional)
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            layers.append(nn.ReLU())

            # Dropout
            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))

            prev_dim = hidden_dim

        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input activations [batch_size, input_dim]

        Returns:
            Predictions [batch_size, 1], values in [0, 1]
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities for factual class."""
        with torch.no_grad():
            return self.forward(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Return binary predictions.

        Args:
            x: Input activations
            threshold: Classification threshold

        Returns:
            Binary predictions (0 or 1)
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).long()

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the predictor
    predictor = EpistemicPredictor(
        input_dim=1536,
        hidden_dims=[512, 256, 128],
        dropout=[0.3, 0.2, 0.0]
    )

    print(f"Predictor architecture:")
    print(predictor)
    print(f"\nTotal parameters: {predictor.count_parameters():,}")

    # Test forward pass
    batch_size = 32
    dummy_input = torch.randn(batch_size, 1536)
    output = predictor(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"✓ Predictor test passed")
