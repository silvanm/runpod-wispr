# RunPod serverless worker: transcribe MP3/MP4 → CSV with speaker diarization
# GPU image with CUDA for WhisperX; ffmpeg for MP4 audio extraction
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/handler.py /handler.py

ENV PYTHONUNBUFFERED=1
CMD ["python3", "-u", "/handler.py"]
