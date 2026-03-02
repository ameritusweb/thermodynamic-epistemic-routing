"""
1D CNN predictor over all-layer stacked activations.

Treats the stacked per-layer activations as a spatial signal along the depth
axis, allowing the network to detect cross-layer patterns — e.g. the
"blushing zone" rise-peak-fall shape — that a single-layer MLP or simple
feature concatenation cannot capture.

Input shape : [batch, n_layers, hidden_dim]
Output shape: [batch, 1]  — probability in [0, 1]
"""

import torch
import torch.nn as nn


class EpistemicCNNPredictor(nn.Module):
    """
    1D CNN over the transformer-layer dimension.

    Architecture:
        1. Pointwise conv  (kernel_size=1): hidden_dim → channel_dim
           Independently projects each layer's feature vector; no cross-layer
           mixing yet. Acts like a shared MLP applied to every row of the image.

        2. Spatial convs   (kernel_size=3): learn local cross-layer patterns
           Each output position attends to a window of 3 consecutive layers,
           detecting motifs like "signal rising here" or "peak at this depth".

        3. Global avg-pool + global max-pool, concatenated → [batch, channel_dim]
           avg captures the mean activation level; max captures the peak response.

        4. FC classifier → [batch, 1]
    """

    def __init__(
        self,
        n_layers: int = 28,
        hidden_dim: int = 1536,
        channel_dim: int = 128,
        dropout: float = 0.3,
        conv_dropout: float = 0.0,
    ):
        """
        Args:
            n_layers: Number of transformer layers (spatial / sequence dimension).
            hidden_dim: Hidden state size of the base model (or PCA output dim).
            channel_dim: Internal channel width after pointwise compression.
                         Total params ≈ hidden_dim × channel_dim (pointwise conv dominates).
                         128 → ~200k params on Qwen2.5-1.5B; safe for 16k samples.
            dropout: Dropout probability in the FC classifier head.
            conv_dropout: Dropout1d probability applied after each spatial conv.
                          Drops entire channel maps, more effective than scalar dropout
                          for convolutional layers. 0.0 = disabled.
        """
        super().__init__()
        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim
        self.channel_dim = channel_dim

        def _conv_block(in_ch, out_ch, kernel_size, padding=0):
            """Conv → BN → ReLU → optional Dropout1d."""
            layers = [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if conv_dropout > 0.0:
                layers.append(nn.Dropout1d(p=conv_dropout))
            return layers

        # Step 1 — pointwise compression: hidden_dim → channel_dim
        # bias=False because BatchNorm absorbs the bias
        self.channel_reduce = nn.Sequential(
            *_conv_block(hidden_dim, channel_dim, kernel_size=1),
        )

        # Step 2 — spatial convs along the layer axis
        # Three layers: channel_dim → channel_dim → channel_dim → channel_dim // 2
        # Padding=1 preserves the spatial length (n_layers stays unchanged).
        half = channel_dim // 2
        self.spatial_convs = nn.Sequential(
            *_conv_block(channel_dim, channel_dim, kernel_size=3, padding=1),
            *_conv_block(channel_dim, channel_dim, kernel_size=3, padding=1),
            *_conv_block(channel_dim, half,         kernel_size=3, padding=1),
        )

        # Step 3 — global pool: avg + max → [batch, channel_dim]
        # (half * 2 = channel_dim)

        # Step 4 — classifier
        self.classifier = nn.Sequential(
            nn.Linear(channel_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_layers, hidden_dim]

        Returns:
            [batch, 1] predictions in [0, 1]
        """
        # Conv1d expects [batch, channels, length]
        x = x.transpose(1, 2)           # [batch, hidden_dim, n_layers]
        x = self.channel_reduce(x)       # [batch, channel_dim, n_layers]
        x = self.spatial_convs(x)        # [batch, channel_dim//2, n_layers]

        # Global pooling — avg and max capture complementary information
        avg = x.mean(dim=2)              # [batch, channel_dim//2]
        mx  = x.max(dim=2).values        # [batch, channel_dim//2]
        x   = torch.cat([avg, mx], dim=1)  # [batch, channel_dim]

        return self.classifier(x)        # [batch, 1]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
