# Neural Pathway Routing PoC - RunPod Dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace:$PYTHONPATH \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    htop \
    tmux \
    vim \
    nano \
    tree \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set up workspace
WORKDIR /workspace

# Create necessary directories
RUN mkdir -p /workspace/data/raw \
    /workspace/data/processed \
    /workspace/data/activations \
    /workspace/data/splits \
    /workspace/outputs/checkpoints \
    /workspace/outputs/logs \
    /workspace/outputs/visualizations \
    /workspace/outputs/metrics \
    /workspace/outputs/artifacts \
    /workspace/.cache/huggingface

# Copy entrypoint script
COPY runpod_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
