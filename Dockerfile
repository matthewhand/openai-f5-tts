# Use a lightweight Python 3.10 base image
FROM python:3.10-slim AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    WORKDIR=/app \
    REF_AUDIO_DIR=/app/ref_audio \
    CKPTS_DIR=/app/ckpts \
    CACHE_DIR=/app/cache \
    HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch \
    PYTHONPATH=${WORKDIR}

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

# Copy full project into working directory (app folder preserved)
COPY . ${WORKDIR}/

# Install Python dependencies via setup.py
RUN python3 -m pip install --no-cache-dir .

# Allow optional override to skip downloading assets and mount existing ckpts dir
VOLUME ["${CKPTS_DIR}", "${CACHE_DIR}"]

# Optional asset downloads (reference audio)
ARG SKIP_DOWNLOAD_ASSETS=false
RUN if [ "${SKIP_DOWNLOAD_ASSETS}" = "false" ]; then \
      curl -L -o ${REF_AUDIO_DIR}/basic_ref_en.wav \
        https://github.com/SWivid/F5-TTS/raw/refs/heads/main/src/f5_tts/infer/examples/basic/basic_ref_en.wav; \
    else \
      echo "Skipping asset downloads"; \
    fi

# Default command to run the server
CMD ["python", "-m", "app.cli", "--host", "0.0.0.0", "--port", "9090"]
