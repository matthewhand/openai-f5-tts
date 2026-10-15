# Pin bookworm: python:3.10-slim tracks trixie, and ffmpeg there pulls llvm/mesa
FROM python:3.10-slim-bookworm AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    WORKDIR=/app \
    REF_AUDIO_DIR=/app/ref_audio \
    CKPTS_DIR=/app/ckpts \
    CACHE_DIR=/app/ref_audio_cache

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    -o Acquire::Retries=5 \
    curl \
    git \
    sox \
    libsox-fmt-mp3 \
    libsndfile1 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR ${WORKDIR}

# Create expected directories
RUN mkdir -p ${REF_AUDIO_DIR} ${CKPTS_DIR} ${CACHE_DIR}

# Copy Python dependencies
COPY requirements.txt ${WORKDIR}/

# Install Python dependencies, then replace torch with the CUDA wheel.
# extra-index-url keeps PyPI for the other packages.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# Copy application code
COPY app/ ${WORKDIR}/

# Reference audio is provided by the operator's volume at runtime.
# Never fetch unverified voice content implicitly during image builds.

# Default command to run the server
CMD ["python3", "/app/server.py"]
