"""
Multi-layer, multi-token activation extractor.

Hooks specified layer indices simultaneously (single forward pass per batch),
captures the last N non-padding tokens from each layer, mean-pools them,
then concatenates across layers to produce a richer feature vector.

Output shape: [batch, len(layer_indices) * hidden_dim]
"""

import torch
import logging
from typing import List, Optional
from transformers import PreTrainedTokenizer


class MultiFeatureExtractor:
    """
    Extract mean-pooled last-N-token activations from multiple transformer layers.

    Usage:
        extractor = MultiFeatureExtractor(model, layer_indices=[14, 20, 26], n_tokens=5)
        features = extractor.extract_from_texts(texts, tokenizer, device=device)
        # features.shape == [len(texts), len(layer_indices) * hidden_dim]
    """

    def __init__(self, model, layer_indices: List[int], n_tokens: int = 1):
        """
        Args:
            model: HuggingFace model (plain or PEFT-wrapped)
            layer_indices: Which layers to extract from (e.g. [14, 20, 26]).
                           Negative indices supported (e.g. -2 = penultimate).
            n_tokens: How many trailing non-padding tokens to mean-pool per layer.
                      1 = last token only (equivalent to ActivationExtractor).
        """
        self.model = model
        self.n_tokens = n_tokens
        self.hooks: list = []
        self._attention_mask: Optional[torch.Tensor] = None
        self._layer_activations: dict = {}   # {absolute_idx: [batch, seq, hidden]}

        # Resolve absolute indices once
        layers = self._resolve_layers()
        n_total = len(layers)
        self.abs_indices = [i % n_total for i in layer_indices]
        self._layers = layers
        logging.debug(
            f"MultiFeatureExtractor: {n_total} total layers, "
            f"extracting from {self.abs_indices}, n_tokens={n_tokens}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_layers(self):
        import torch.nn as nn
        import logging
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

    def _make_hook(self, abs_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Store full [batch, seq, hidden] — we reduce per-sample below in extract()
            self._layer_activations[abs_idx] = hidden.detach()
        return hook

    def _pool_tokens(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Mean-pool the last `n_tokens` non-padding positions for each sample.

        Args:
            hidden: [batch, seq, hidden]
            attention_mask: [batch, seq] or None

        Returns:
            [batch, hidden]
        """
        batch_size = hidden.size(0)

        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).long()   # [batch] — number of real tokens
        else:
            seq_lengths = torch.full((batch_size,), hidden.size(1), dtype=torch.long, device=hidden.device)

        out = []
        for i in range(batch_size):
            last = seq_lengths[i].item()                     # exclusive end
            start = max(0, last - self.n_tokens)             # inclusive start
            token_slice = hidden[i, start:last, :]           # [k, hidden]  k <= n_tokens
            out.append(token_slice.mean(dim=0))              # [hidden]

        return torch.stack(out, dim=0).cpu().float()         # [batch, hidden]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_hooks(self):
        """Attach forward hooks to all target layers."""
        for abs_idx in self.abs_indices:
            h = self._layers[abs_idx].register_forward_hook(self._make_hook(abs_idx))
            self.hooks.append(h)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def extract(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Run one forward pass and return concatenated multi-layer features.

        Args:
            input_ids: [batch, seq]
            attention_mask: [batch, seq] or None

        Returns:
            [batch, len(layer_indices) * hidden_dim]
        """
        if not self.hooks:
            self.register_hooks()

        self._layer_activations.clear()
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Pool each layer and concatenate in the order the user specified
        pooled = []
        for abs_idx in self.abs_indices:
            hidden = self._layer_activations[abs_idx]
            pooled.append(self._pool_tokens(hidden, attention_mask))   # [batch, hidden]

        return torch.cat(pooled, dim=1)   # [batch, len(layers) * hidden]

    def extract_from_texts(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32,
        max_length: int = 2048,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Extract features for a list of texts.

        Args:
            texts: Input strings
            tokenizer: HuggingFace tokenizer
            batch_size: Processing batch size
            max_length: Tokeniser truncation limit
            device: Device to run model on

        Returns:
            [len(texts), len(layer_indices) * hidden_dim]
        """
        from tqdm import tqdm
        self.register_hooks()
        all_features = []

        try:
            for start in tqdm(range(0, len(texts), batch_size), desc="Extracting multi-layer features"):
                batch = texts[start:start + batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                input_ids = enc['input_ids'].to(device)
                attention_mask = enc['attention_mask'].to(device)
                features = self.extract(input_ids, attention_mask)   # [batch, feat_dim]
                all_features.append(features)
        finally:
            self.remove_hooks()

        return torch.cat(all_features, dim=0)   # [n_texts, feat_dim]

    @property
    def feature_dim(self) -> Optional[int]:
        """
        Return expected output feature dimension, or None if not yet known.
        Requires one forward pass to determine hidden_dim.
        """
        # We can infer from the layer's weight if available
        try:
            layer = self._layers[self.abs_indices[0]]
            # Try common attribute names for hidden dimension
            for attr in ['hidden_size', 'embed_dim', 'd_model']:
                if hasattr(layer, attr):
                    return len(self.abs_indices) * getattr(layer, attr)
            # Try reading from weight matrix
            for name, param in layer.named_parameters():
                if 'weight' in name and param.dim() == 2:
                    hidden_dim = max(param.shape)
                    return len(self.abs_indices) * hidden_dim
        except Exception:
            pass
        return None

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
