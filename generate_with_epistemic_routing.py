"""
Thermodynamic Epistemic Routing — Live Generation Demo

Streams output token-by-token with real-time color coding:
  GREEN = factual regime  (T(x) > threshold — model is retrieving)
  RED   = speculative regime (T(x) < threshold — model is confabulating)

T(x) = mean(||h_{n+1} - h_n||) across thermo_layers at each generated token.
Zero additional parameters. The physics does all the work.

Usage:
    python generate_with_epistemic_routing.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # must match training base
LORA_PATH       = "outputs/checkpoints/lora_final"
THERMO_LAYERS   = [16, 17, 18, 19]               # layers from eval
THRESHOLD       = 27.2                            # calibrated on held-out set
MAX_NEW_TOKENS  = 300
TEMPERATURE     = 0.0                             # greedy — deterministic routing demo
# ─────────────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=False,
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# ── Thermodynamic hooks ───────────────────────────────────────────────────────
# Capture last-token hidden state for each targeted layer on every forward pass.

current_h = {}   # layer_idx -> Tensor[batch, hidden]

def make_hook(layer_idx):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        current_h[layer_idx] = hidden[:, -1, :].detach().clone()
    return hook

hooks = []
for idx in THERMO_LAYERS:
    layer = model.base_model.model.layers[idx]
    hooks.append(layer.register_forward_hook(make_hook(idx)))

print(f"Hooks attached to layers {THERMO_LAYERS}")
print(f"Threshold = {THRESHOLD}  |  \033[92mGreen = factual\033[0m  |  \033[91mRed = speculative\033[0m\n")


def compute_tx() -> float:
    """Compute T(x) = mean(||h_{n+1} - h_n||) from hooked layer states."""
    norms = []
    for i in range(len(THERMO_LAYERS) - 1):
        l1, l2 = THERMO_LAYERS[i], THERMO_LAYERS[i + 1]
        if l1 in current_h and l2 in current_h:
            delta = current_h[l2] - current_h[l1]
            norms.append(torch.norm(delta, p=2, dim=-1).item())
    return sum(norms) / len(norms) if norms else 0.0


def generate(prompt: str):
    if not prompt.strip():
        return

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()
    past_key_values = None

    print("\nResponse: ", end="", flush=True)

    for step in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            outputs = model(
                input_ids=generated if step == 0 else generated[:, -1:],
                use_cache=True,
                past_key_values=past_key_values,
            )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        T = compute_tx()

        if TEMPERATURE == 0.0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits / TEMPERATURE, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(0)

        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
        color = "\033[92m" if T > THRESHOLD else "\033[91m"   # green / red
        print(f"{color}{token_text}\033[0m", end="", flush=True)

        # Uncomment to show live T(x) value per token:
        # print(f" \033[90m[T={T:.1f}]\033[0m", end="", flush=True)

        if next_token.item() == tokenizer.eos_token_id:
            break

    print("\n" + "─" * 80)


# ── Interactive loop ──────────────────────────────────────────────────────────
print("Thermodynamic Epistemic Routing — LIVE DEMO")
print("Enter a prompt to see factual (green) vs speculative (red) regimes in real time.")
print("Empty line exits.\n")

try:
    while True:
        prompt = input("Prompt > ").strip()
        if not prompt:
            break
        generate(prompt)
except KeyboardInterrupt:
    print("\n\nExiting...")
finally:
    for h in hooks:
        h.remove()
    print("Hooks removed.")
