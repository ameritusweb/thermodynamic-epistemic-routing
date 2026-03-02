"""LoRA configuration for efficient fine-tuning."""

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import PreTrainedModel
import logging


def get_lora_config(
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    target_modules: list = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> LoraConfig:
    """
    Create LoRA configuration.

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout rate
        target_modules: List of module names to apply LoRA to
        bias: Bias handling ("none", "all", "lora_only")
        task_type: Task type

    Returns:
        LoRA configuration
    """
    if target_modules is None:
        # Default for Qwen/Llama models
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else task_type
    )

    logging.info(f"LoRA Config created:")
    logging.info(f"  Rank: {r}")
    logging.info(f"  Alpha: {lora_alpha}")
    logging.info(f"  Dropout: {lora_dropout}")
    logging.info(f"  Target modules: {target_modules}")

    return config


def apply_lora(model: PreTrainedModel, lora_config: LoraConfig) -> PeftModel:
    """
    Apply LoRA to a model.

    Args:
        model: Base model
        lora_config: LoRA configuration

    Returns:
        Model with LoRA adapters applied
    """
    peft_model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())

    logging.info(f"LoRA applied successfully:")
    logging.info(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logging.info(f"  Total parameters: {total_params:,}")

    return peft_model


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )


if __name__ == "__main__":
    # Test LoRA config
    config = get_lora_config(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1
    )

    print("\n✓ LoRA configuration created successfully")
    print(f"  Config: {config}")
