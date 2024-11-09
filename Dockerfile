# Use PyTorch with CUDA 11.8 for compatibility with the GTX 1080
FROM nvcr.io/nvidia/pytorch:24.10-py3
USER root

# Install essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg python3-pip \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Clone and install F5-TTS repository
WORKDIR /workspace
RUN git clone https://github.com/SWivid/F5-TTS.git \
    && cd F5-TTS \
    && pip install -e .[eval]

# Copy app-specific requirements and files
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt

# Copy the app directory into the container
COPY app/ /app

# Set the default command to run the server
CMD ["python", "/app/server.py"]

