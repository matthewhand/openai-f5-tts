# Pinned multi-architecture Python 3.10 Bookworm image index; this service builds
# only for linux/amd64. Inspect the digest again before a planned base update.
FROM python:3.10-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370 AS base

ARG TARGETARCH
# CPU is the source-level default. The NVIDIA Compose profile passes cu124.
ARG TORCH_CHANNEL=cpu

ENV DEBIAN_FRONTEND=noninteractive \
    WORKDIR=/app \
    REF_AUDIO_DIR=/app/ref_audio \
    CKPTS_DIR=/app/ckpts \
    CACHE_DIR=/app/ref_audio_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    -o Acquire::Retries=5 \
    curl \
    git \
    sox \
    libsox-fmt-mp3 \
    libsndfile1 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${WORKDIR}
RUN mkdir -p ${REF_AUDIO_DIR} ${CKPTS_DIR} ${CACHE_DIR} /app/constraints

COPY requirements.txt /app/requirements.txt
COPY constraints/py310-torch25.txt /app/constraints/py310-torch25.txt

# Fail closed on unknown channel/architecture. Install exactly one Torch flavor
# from its own index before transitive dependencies; no PyPI fallback for wheels.
# The constraints must remain active while resolving all F5-TTS dependencies.
RUN set -eu; \
    test "$TARGETARCH" = amd64; \
    case "$TORCH_CHANNEL" in cpu|cu124) ;; *) echo 'Unsupported TORCH_CHANNEL' >&2; exit 64;; esac; \
    python3 -m pip install --no-cache-dir pip==25.2; \
    python3 -m pip install --no-cache-dir --no-deps \
      --index-url "https://download.pytorch.org/whl/$TORCH_CHANNEL" \
      "torch==2.5.1+$TORCH_CHANNEL" "torchaudio==2.5.1+$TORCH_CHANNEL"; \
    python3 -m pip install --no-cache-dir --index-url https://pypi.org/simple \
      -c constraints/py310-torch25.txt -r requirements.txt; \
    python3 -m pip check; \
    python3 -c 'import importlib.metadata as m, os; v="2.5.1+"+os.environ["TORCH_CHANNEL"]; assert m.version("torch")==v, (v,m.version("torch")); assert m.version("torchaudio")==v, (v,m.version("torchaudio"))'

COPY app/ ${WORKDIR}/

# Operator-provided checkpoints and reference audio are mounted at runtime.
# Never put model weights, voice samples, or host secrets in the image.
CMD ["python3", "/app/server.py"]
