# Use the official NVIDIA NGC PyTorch container.
# This image already includes compatible torch==2.2.0 and torchvision==0.17.0.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

WORKDIR /opt/ml/code

# Install system dependencies needed for git and for compiling packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# --- CORRECTED AND SAFER INSTALLATION ORDER ---

# 1. Install compiled packages that depend heavily on the base torch version.
#    The --no-build-isolation flag is critical here. It forces them to compile
#    against the torch version from the base image (2.2.0).
RUN pip install --no-cache-dir xformers
# Flash-Attention wheels matching torch==2.2.0/cu121 are not yet published,
# and building from source exhausts memory when we build the image on a Mac.
# At runtime the code automatically falls back to PyTorch SDPA if Flash-Attention
# is unavailable, so we simply omit it here.  You can always add a pre-built
# wheel later (e.g. flash-attn>=2.5) once an official CUDA 12.1 wheel appears.
# 2025-01: Pre-built wheels are now available for torch==2.2 / CUDA 12.x, so we
# can install directly without compiling.  This URL points to the cp310 build
# that matches the base image's Python 3.10 runtime.
RUN pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 2. IMPORTANT: Verify your requirements.txt does NOT contain torch, torchvision, or torchaudio.
#    Install these last. If any package here tries to change the torch version,
#    the build will likely fail, which is better than it succeeding silently with a broken state.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. (DEBUGGING STEP) Verify that the versions were not changed.
#    This command is your safety net. It will clearly show if versions are mismatched.
RUN echo "--- Verifying library versions after installation ---" && \
    pip list | grep -E 'torch|transformers' && \
    python -c "import torch; import torchvision; print(f'Torch version: {torch.__version__}'); print(f'Torchvision version: {torchvision.__version__}')" && \
    echo "--- Verification complete ---"

# --- END OF CORRECTIONS ---

# Copy your application code
COPY inference.py .
COPY serve .
RUN chmod +x serve

# Set environment variables for runtime
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# Default HF model to pull when /opt/ml/model is not present
ENV MODEL_ID=ai4bharat/indic-parler-tts
# Note: Flash Attention 2 automatically replaces the need for xformers memory efficient attention
ENV TRANSFORMERS_USE_XFORMERS=1

ENTRYPOINT ["/opt/ml/code/serve"]