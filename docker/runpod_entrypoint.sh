#!/bin/bash
# Neural Pathway Routing PoC - RunPod Entrypoint Script

set -e

echo "=================================="
echo "Neural Pathway Routing PoC - RunPod"
echo "=================================="

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

# Set up git (if not already configured)
if [ ! -f ~/.gitconfig ]; then
    git config --global user.email "runpod@example.com"
    git config --global user.name "RunPod User"
fi

# Initialize wandb (if API key is set)
if [ -n "$WANDB_API_KEY" ]; then
    echo "Weights & Biases API key detected"
    wandb login $WANDB_API_KEY
fi

# Print Python and PyTorch info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
fi

echo ""
echo "Environment ready!"
echo "Workspace: /workspace"
echo ""

# Execute the command passed to docker run
exec "$@"
