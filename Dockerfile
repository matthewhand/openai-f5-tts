# Use a lightweight Python 3.10 base image
FROM python:3.10-slim AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    WORKDIR=/app \
    REF_AUDIO_DIR=/app/ref_audio \
    CKPTS_DIR=/app/ckpts \
    CACHE_DIR=/app/ref_audio_cache

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    sox \
    libsox-fmt-mp3 \
    libsndfile1-dev \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR ${WORKDIR}

# Create expected directories
RUN mkdir -p ${REF_AUDIO_DIR} ${CKPTS_DIR} ${CACHE_DIR}

# Copy Python dependencies
COPY requirements.txt ${WORKDIR}/

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ${WORKDIR}/

# Download example reference audio
RUN curl -L -o ${REF_AUDIO_DIR}/basic_ref_en.wav \
    https://github.com/SWivid/F5-TTS/raw/refs/heads/main/src/f5_tts/infer/examples/basic/basic_ref_en.wav

# Default command to run the server
CMD ["python3", "/app/server.py"]
