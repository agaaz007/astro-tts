FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
WORKDIR /opt/ml/code

COPY requirements.txt .
RUN apt-get update && apt-get install -y git build-essential && pip install --no-cache-dir -r requirements.txt

COPY inference.py .
COPY download_model.py .
# ENV TORCH_LOGS="+dynamo"
# ENV TORCHDYNAMO_VERBOSE=1
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV HUGGINGFACE_HUB_TOKEN=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
# Optional NVIDIA envs for extra compatibility
# ENV NVIDIA_VISIBLE_DEVICES=all
# ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# ENV NVIDIA_REQUIRE_CUDA="cuda>=11.8" # Match base image CUDA version

ENTRYPOINT ["python", "inference.py"]