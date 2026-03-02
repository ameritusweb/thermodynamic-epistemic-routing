"""Activation extraction utilities for LLMs."""

import torch
import torch.nn as nn
import logging
from typing import Optional, List, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer


class ActivationExtractor:
    """
    Extract activation vectors from specific layers of a transformer model.

    Uses PyTorch hooks to capture intermediate activations during forward pass.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        layer_index: int = -2,  # Penultimate layer by default
        position: str = "last"  # "last", "first", or "mean"
    ):
        """
        Initialize activation extractor.

        Args:
            model: HuggingFace transformer model
            layer_index: Which layer to extract from (-2 = penultimate)
            position: Which token position to extract ("last", "first", "mean")
        """
        self.model = model
        self.layer_index = layer_index
        self.position = position
        self.activations = None
        self.hook_handle = None

    def _get_activation_hook(self):
        """Create hook function to capture activations."""
        def hook(module, input, output):
            # output is typically a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            self.activations = hidden_states.detach()

        return hook

    def register_hook(self):
        """Register forward hook on target layer."""
        # Unwrap PEFT/LoRA if present: PeftModel.base_model (LoraModel) .model (CausalLM)
        # Plain HuggingFace models have base_model returning self — check it's actually different
        m = self.model
        if hasattr(m, 'base_model') and m.base_model is not m and hasattr(m.base_model, 'model'):
            m = m.base_model.model  # LoraModel.model = e.g. Qwen2ForCausalLM

        # Now resolve layers from the (possibly unwrapped) model:
        # 1. VLM style (Gemma3ForConditionalGeneration): m.language_model.model.layers
        # 2. CausalLM style (Qwen2ForCausalLM): m.model.layers
        # 3. Bare model style (Qwen2Model): m.layers
        # 4. GPT-2 style: m.transformer.h
        if hasattr(m, 'language_model'):
            lm = m.language_model
            if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
                layers = lm.model.layers
            elif hasattr(lm, 'layers'):
                layers = lm.layers
            else:
                layers = None
        elif hasattr(m, 'model') and hasattr(m.model, 'layers'):
            layers = m.model.layers
        elif hasattr(m, 'layers'):
            layers = m.layers
        elif hasattr(m, 'transformer') and hasattr(m.transformer, 'h'):
            layers = m.transformer.h
        else:
            layers = None

        if layers is None:
            import torch.nn as nn
            layers = None
            for name, module in m.named_modules():
                if isinstance(module, nn.ModuleList) and len(module) >= 10:
                    first = module[0]
                    if any(hasattr(first, a) for a in ('self_attn', 'attention', 'attn', 'self_attention')):
                        if layers is None or len(module) > len(layers):
                            layers = module
            if layers is None:
                raise ValueError(
                    f"Could not find model layers in {type(m).__name__}. "
                    f"Top-level children: {[n for n, _ in m.named_children()]}"
                )

        target_layer = layers[self.layer_index]
        self.hook_handle = target_layer.register_forward_hook(self._get_activation_hook())
        logging.debug(f"Hook registered on layer {self.layer_index}")

    def remove_hook(self):
        """Remove the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            logging.debug("Hook removed")

    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract activations for given inputs.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Activation vectors [batch_size, hidden_dim]
        """
        # Register hook if not already registered
        if self.hook_handle is None:
            self.register_hook()

        # Forward pass (activations captured by hook)
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract activation at specified position
        activations = self.activations  # [batch_size, seq_len, hidden_dim]

        if self.position == "last":
            # Get last non-padding token for each sequence
            if attention_mask is not None:
                # Find last non-padding position for each sequence
                seq_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
                batch_indices = torch.arange(activations.size(0), device=activations.device)
                activation_vectors = activations[batch_indices, seq_lengths]  # [batch_size, hidden_dim]
            else:
                # Just take last position
                activation_vectors = activations[:, -1, :]  # [batch_size, hidden_dim]

        elif self.position == "first":
            activation_vectors = activations[:, 0, :]  # [batch_size, hidden_dim]

        elif self.position == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(activations.size()).float()
                sum_activations = torch.sum(activations * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                activation_vectors = sum_activations / sum_mask
            else:
                activation_vectors = activations.mean(dim=1)

        else:
            raise ValueError(f"Invalid position: {self.position}")

        return activation_vectors

    def extract_from_texts(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32,
        max_length: int = 2048,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Extract activations from list of texts.

        Args:
            texts: List of input texts
            tokenizer: HuggingFace tokenizer
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            device: Device to run on

        Returns:
            Activation vectors [num_texts, hidden_dim]
        """
        all_activations = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            # Extract activations
            activations = self.extract(input_ids, attention_mask)
            all_activations.append(activations.cpu())

        # Concatenate all batches
        return torch.cat(all_activations, dim=0)

    def __enter__(self):
        """Context manager entry."""
        self.register_hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.remove_hook()


if __name__ == "__main__":
    # Test the extractor
    from transformers import AutoModel, AutoTokenizer

    print("Testing ActivationExtractor...")

    # Load a small model for testing
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Create extractor
    extractor = ActivationExtractor(model, layer_index=-2, position="last")

    # Test texts
    texts = [
        "Paris is the capital of France.",
        "The sky is blue because of Rayleigh scattering."
    ]

    # Extract activations
    print("Extracting activations...")
    activations = extractor.extract_from_texts(
        texts,
        tokenizer,
        batch_size=2,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"✓ Extraction successful")
    print(f"  Activations shape: {activations.shape}")
    print(f"  Hidden dimension: {activations.shape[1]}")

    extractor.remove_hook()
