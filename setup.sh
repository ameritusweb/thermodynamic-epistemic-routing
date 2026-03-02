#!/bin/bash
# Neural Pathway Routing PoC - Setup Script for RunPod

set -e

echo "=================================="
echo "Neural Pathway Routing PoC Setup"
echo "=================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: No GPU detected"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{raw,processed,activations,splits}
mkdir -p outputs/{checkpoints,logs,visualizations,metrics,artifacts}

# Set up environment file
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API keys:"
    echo "   nano .env"
    echo ""
fi

# Test imports
echo "Testing imports..."
python3 -c "
import torch
import transformers
import peft
from anthropic import Anthropic
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ Transformers {transformers.__version__}')
print(f'✓ PEFT {peft.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=================================="
echo "✓ Setup complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. (Optional) Edit config/base_config.yaml to adjust parameters"
echo "3. Run: python main.py --phase all"
echo ""
