# Base image with CUDA runtime for GPU-accelerated training/inference.
# Falls back to CPU execution automatically if no GPU is present at runtime
# (see src/segmentation/train.py device selection), but training 3D
# volumes on CPU is impractically slow - see README GPU requirements.
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip python3.11-venv \
        libgl1 libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
