FROM python:3.10-slim AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends wget curl git sox libsox-fmt-mp3 libsndfile1-dev ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY app/ /app
RUN curl -L -O https://github.com/SWivid/F5-TTS/raw/refs/heads/main/src/f5_tts/infer/examples/basic/basic_ref_en.wav
CMD ["python3", "/app/server.py"]
