"""
Stacked all-layer activation extractor.

Hooks every transformer layer simultaneously in a single forward pass.
Each hook immediately mean-pools the last N non-padding tokens and moves
the result to CPU, so peak extra GPU memory is O(batch × hidden) per layer
rather than O(batch × seq_len × hidden) per layer.

Output shape: [n_examples, n_layers, hidden_dim]
"""

import torch
import logging
from typing import List, Optional
from transformers import PreTrainedTokenizer


class StackedLayerExtractor:
    """
    Extract mean-pooled last-N-token activations from all transformer layers.

    Compared with MultiFeatureExtractor:
    - Returns [n, n_layers, hidden_dim] (stacked) rather than [n, n_layers * hidden_dim] (concatenated)
    - Reduces each layer to [batch, hidden_dim] inside the hook, so the full
      [batch, seq, hidden] tensor is never retained for all layers simultaneously
    - Suitable for feeding into a CNN that treats the layer axis as spatial

    Usage:
        extractor = StackedLayerExtractor(model, n_tokens=5)
        features = extractor.extract_from_texts(texts, tokenizer, device=device)
        # features.shape == [len(texts), n_layers, hidden_dim]
    """

    def __init__(self, model, n_tokens: int = 5):
        """
        Args:
            model: HuggingFace model (plain or PEFT-wrapped)
            n_tokens: How many trailing non-padding tokens to mean-pool per layer.
        """
        self.model = model
        self.n_tokens = n_tokens
        self.hooks: list = []
        self._attention_mask: Optional[torch.Tensor] = None
        self._activations: dict = {}   # {layer_idx: [batch, hidden_dim]}  already reduced

        layers = self._resolve_layers()
        self._layers = layers
        self.n_layers = len(layers)
        logging.debug(f"StackedLayerExtractor: {self.n_layers} layers, n_tokens={n_tokens}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_layers(self):
        import torch.nn as nn
        m = self.model
        if hasattr(m, 'base_model') and m.base_model is not m and hasattr(m.base_model, 'model'):
            m = m.base_model.model
        if hasattr(m, 'language_model'):
            lm = m.language_model
            if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
                return lm.model.layers
            if hasattr(lm, 'layers'):
                return lm.layers
        if hasattr(m, 'model') and hasattr(m.model, 'layers'):
            return m.model.layers
        if hasattr(m, 'layers'):
            return m.layers
        if hasattr(m, 'transformer') and hasattr(m.transformer, 'h'):
            return m.transformer.h
        best = None
        for name, module in m.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) >= 10:
                first = module[0]
                if any(hasattr(first, a) for a in ('self_attn', 'attention', 'attn', 'self_attention')):
                    if best is None or len(module) > len(best):
                        best = module
        if best is not None:
            return best
        raise ValueError(
            f"Cannot find transformer layers in {type(self.model).__name__}. "
            f"Top-level children: {[n for n, _ in self.model.named_children()]}"
        )

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden: [batch, seq, hidden_dim]  on GPU

            mask = self._attention_mask
            batch_size = hidden.size(0)

            if mask is not None:
                seq_lengths = mask.sum(dim=1).long()   # [batch], on same device as mask
            else:
                seq_lengths = torch.full(
                    (batch_size,), hidden.size(1),
                    dtype=torch.long, device=hidden.device
                )

            # Reduce to [batch, hidden_dim] immediately — don't retain full seq dim
            out = []
            for i in range(batch_size):
                last  = seq_lengths[i].item()
                start = max(0, last - self.n_tokens)
                # Slice stays on GPU; mean collapses seq → scalar per feature
                out.append(hidden[i, start:last, :].mean(dim=0))   # [hidden_dim]

            # Move to CPU as float32 so GPU memory is freed promptly
            self._activations[layer_idx] = torch.stack(out).detach().cpu().float()

        return hook

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_hooks(self):
        """Attach forward hooks to all transformer layers."""
        for i, layer in enumerate(self._layers):
            h = layer.register_forward_hook(self._make_hook(i))
            self.hooks.append(h)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Run one forward pass and return stacked all-layer features.

        Args:
            input_ids: [batch, seq]
            attention_mask: [batch, seq] or None

        Returns:
            [batch, n_layers, hidden_dim]
        """
        if not self.hooks:
            self.register_hooks()

        self._attention_mask = attention_mask
        self._activations.clear()

        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Stack layers in ascending order: [batch, n_layers, hidden_dim]
        layer_tensors = [self._activations[i] for i in range(self.n_layers)]
        return torch.stack(layer_tensors, dim=1)   # [batch, n_layers, hidden]

    def extract_from_texts(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32,
        max_length: int = 2048,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Extract stacked features for a list of texts.

        Args:
            texts: Input strings
            tokenizer: HuggingFace tokenizer
            batch_size: Processing batch size
            max_length: Tokenizer truncation limit
            device: Device to run model on

        Returns:
            [len(texts), n_layers, hidden_dim]
        """
        from tqdm import tqdm
        self.register_hooks()
        all_features = []

        try:
            for start in tqdm(range(0, len(texts), batch_size), desc="Extracting all-layer features"):
                batch = texts[start:start + batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                input_ids     = enc['input_ids'].to(device)
                attention_mask = enc['attention_mask'].to(device)
                features = self.extract(input_ids, attention_mask)   # [batch, n_layers, hidden]
                all_features.append(features)
        finally:
            self.remove_hooks()

        return torch.cat(all_features, dim=0)   # [n_texts, n_layers, hidden_dim]
