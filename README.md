# openai-f5-tts

<img src="assets/images/logo.webp" alt="Project Logo" width="200"/>

**Release status: experimental / alpha (source-reviewed 14 October 2026).** This Flask/Gevent adapter exposes local F5-TTS synthesis through an OpenAI-style speech route, lists installed voices/models and supports an optional ElevenLabs-shaped timestamp response. Actual model inference, fresh-install reliability, CPU use and Docker deployment have **not** been revalidated for this revision. Do not expose this service to the public Internet.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment Instructions](#deployment-instructions)
  - [Using Docker Compose (Experimental Local Preview)](#using-docker-compose-experimental-local-preview)
- [Development Setup](#development-setup)
  - [Prerequisites](#prerequisites-1)
  - [Installation Steps](#installation-steps)
- [API Endpoints](#api-endpoints)
- [Adding Your Own Fine-Tuned Checkpoint](#adding-your-own-fine-tuned-checkpoint)
- [TODO](#todo)
- [Support](#support)
- [Contributing](#contributing)
- [License](#license)

---

## Current implementation and limitations

- `POST /v1/audio/speech` generates audio using an installed F5-TTS checkpoint, a matching reference-audio file and a locally loaded Whisper transcription model. The documented `speed` parameter reaches the inference call; supported conversions use FFmpeg.
- `GET /v1/models`, `GET /v1/voices`, `GET /v1/voices/all` and `GET /v1/loaded_models` are authenticated. Voice language metadata is **not** implemented; do not rely on language filtering.
- `POST /v1/text-to-speech/{voice}/with-timestamps` returns base64 audio and, where duration can be read, **duration-weighted estimated timings**, *not* forced alignment. Null alignment is possible.
- Sentence streaming is **not implemented on `main`**; see the proposed [streaming PR #6](https://github.com/matthewhand/openai-f5-tts/pull/6).
- The Compose setup currently requires NVIDIA GPU container support and mounts local checkpoint and reference-audio folders. The installer pulls an **unpinned** upstream F5-TTS dependency; reproducible builds have not been demonstrated.
- **Checkpoint path-trust guard (source-only verified):** discovery now checks that the configured checkpoint root and its ancestors are not symlinks, and that voice directories and selected `.safetensors` files are real directories / singly linked regular files, not symlinks or hardlinks. Immediately before loading a previously discovered voice, the path is checked again **before any loaded model is evicted**. A stale or substituted path fails without invoking the model loader. Keep the checkpoint tree trusted and immutable to less-privileged writers: pathname checks cannot eliminate **replacement races** between validation and the upstream loader opening the file. This does **not** prove file authenticity, content format, signature, integrity, permission confinement or licensing. Live inference and Windows-native path behavior are unverified.
- **Checkpoint loading is native-safetensors-only.** The wrapper discovers `model_1250000.safetensors` first, or `model_1200000.safetensors`; it does **not** deserialize or automatically convert legacy pickle-based `.pt` files. Users with only `.pt` checkpoints must obtain a genuine `.safetensors` export from a trusted offline training/export environment before using this branch. Do not simply rename a `.pt` file.
- Incoming JSON requests are bounded by `MAX_REQUEST_BYTES` (default 65536) and `MAX_TEXT_CHARS` (default 2000); speed must be between 0.25 and 4. A **process-local, non-blocking single-generation gate** now covers model inference and audio conversion. A second simultaneous request receives HTTP **503** rather than waiting behind a growing queue. This is **not** a cross-process or cluster-wide limit. HTTP/body/credential debug logging has been removed, but upstream library logs are not independently audited.
- The Python wrapper is MIT-licensed. **Upstream pretrained F5-TTS model weights are CC-BY-NC** per [SWivid/F5-TTS licensing](https://github.com/SWivid/F5-TTS#license); this repository's MIT license does not grant commercial rights to those weights. Check any reference-audio and voice rights separately.


### Reference-audio and cache pathname trust

- **Reference-audio file trust (source-only guarded):** `get_ref_audio_path` validates a one-segment voice name and a non-symlinked reference-audio root (including ancestors), and accepts only singly linked regular input files with supported extensions. It rejects symlink and hardlink inputs rather than following them. Before FFmpeg writes or reads, cached WAV/PCM paths are checked for real singly linked regular files under a non-symlinked cache root; dangling symlinks, directories and unexpected paths are rejected. Pre-existing intermediate `<voice>_processed.pcm` files now fail closed rather than being clobbered or deleted; an operator must inspect/remove stale files separately in their trusted private workspace. The cached WAV is rechecked before decoding/resampling, and before Whisper reads it. Directory owners must keep `ref_audio/`, `ref_audio_cache/` and their ancestors private to trusted writers. **Metadata checks are not race-free:** a concurrent pathname swap before FFmpeg/Whisper opens the path can defeat these checks. Neither extension nor metadata validation proves audio content, consent, integrity, or provenance, or sandboxes FFmpeg. No live FFmpeg, Whisper or model execution was performed.

### Bounded output buffering and FFmpeg safeguards

- `MAX_AUDIO_OUTPUT_BYTES` (positive integer, default **16777216 = 16 MiB**) bounds each file read by **limit + 1 byte** in both `/v1/audio/speech` and the timestamped base64 route. Oversized generated output returns HTTP **503** with `Generated audio exceeds configured output limit`; the request-owned output file is removed. The size check happens **after** synthesis and encoding, so it does not prevent large intermediate audio allocations or disk writes. Even an accepted timestamped response expands during base64 encoding and JSON serialization; aggregate request memory is not independently bounded.
- `FFMPEG_TIMEOUT_SECONDS` (positive integer, default **120**) applies separately to all **four** FFmpeg call sites: three reference-audio preparation stages and one final audio conversion. FFmpeg is invoked with `-nostdin -hide_banner -loglevel error -nostats`. On reference conversion failure or timeout, incomplete cached WAV and intermediate PCM files are discarded so a later request can retry. A request whose final encoding fails uses the existing per-request audio cleanup; a model inference stall is **not** bounded by the FFmpeg timeout.
- These are **source-level mitigations**, verified through deterministic mocked file/subprocess and real Flask test-client regressions, **not** runtime guarantees for GPU memory, total disk usage, concurrent HTTP responses, native-library termination or multi-worker deployments. No model load, real FFmpeg or container startup was performed in this review.

### Model-cache recovery and authentication contract

- The model cache is a **bounded in-process LRU**, not persistent storage. A voice switch may evict the least recently used model *before* allocating the next model to limit peak memory; if a new checkpoint fails to load, the evicted model is **not automatically restored**. A later authorized request can reload it from disk. Never interpret discovery or a `loaded` flag as successful speech generation.
- A failed load must leave **no partially registered model** in `loaded_models`. `MODEL_LOAD_LIMIT` must be a positive integer; a bad configuration should fail during startup instead of failing mid-request.
- On a failed checkpoint load (including a `None` result), the handler now attempts **best-effort** Python garbage collection and CUDA allocator cache release before re-raising the *original* error. Cleanup errors cannot hide that loading failure. The evicted voice is **not automatically restored**; subsequent authorized requests can retry. Synthetic CPU/CUDA mocks cover failure, retry and secondary cleanup exceptions, but **no live GPU memory-recovery guarantee exists**.
- The published `/v1/models`, `/v1/voices`, `/v1/voices/all`, `/v1/loaded_models`, `/v1/audio/speech`, and `/v1/text-to-speech/{voice}/with-timestamps` routes require `Authorization: Bearer <key>` or `xi-api-key: <key>` when `REQUIRE_API_KEY=True`. Missing or invalid keys should return **401**, including malformed/non-ASCII token values; no secret should be echoed into logs or responses.
- When `REQUIRE_API_KEY=True` (default), a non-placeholder API key is **required at import/startup** and all six HTTP routes enforce Bearer or `xi-api-key` authentication. Setting `REQUIRE_API_KEY=False` is intended **only** for explicit numeric loopback binding (`HOST=127.0.0.1` / `--host 127.0.0.1`, or `::1` on IPv6). A wildcard `0.0.0.0` / `::`, LAN IP, DNS hostname including `localhost`, or any ambiguous address must cause a startup error **before** TTS handler construction. This is a process bind safeguard, not reverse-proxy or host-network isolation; do not expose unauthenticated mode beyond your own machine.
- The offline Flask test-client suite exercises the **actual Flask routes and authentication decorator** while replacing Gevent and model startup with inert fixtures. It verifies Bearer and `xi-api-key` headers, missing/invalid/non-ASCII key rejection (401), explicit optional authentication bypass **without an API key**, enabled-mode placeholder-key rejection, invalid bodies (400), oversized bodies (413) and synthetic busy responses (503). The fixture's busy exception is tied to the same class imported by the server, eliminating a standalone-versus-full-suite import-order false failure. No socket or inference is used; production reverse-proxy behavior remains unverified.

### Concurrency and temporary-file status

The draft branch now rejects simultaneous synthesis work with HTTP 503 inside one process. A separate request-owned encoder removes partial WAV/converted files after success or failure; deterministic mock-backed regressions exercise both paths. The Python server still has **no cross-process admission mechanism**, no proved scheduling fairness and no model-backed concurrent load measurements. Avoid multiple workers or exposing the API on untrusted networks until an independent release review.

## Prerequisites

Before getting started, ensure you have the following installed on your system:

- **Git**: For cloning the repository.
- **Docker and NVIDIA Container Toolkit**: Required for the existing GPU-focused Docker Compose configuration. The published container path has not been freshly verified.
  - **Linux Users**: Tested with `docker.io` on Ubuntu 24.04 LTS.
  - **Windows Users**: Tested with Docker Desktop on Windows 11 (using WSL2 backend).
- **Python 3.10**: Required for development setup.
- **Conda** (optional but recommended for creating an isolated Python environment).

---

## Deployment Instructions

### Using Docker Compose (Experimental Local Preview)

This experimental Compose configuration packages the Flask application and its dependencies; it has **not** passed a clean build or real speech acceptance test. It requires a prepared, legally usable checkpoint and audio sample before it is functional.

**Step 1: Ensure Docker is Installed**

Make sure Docker is installed on your system. You can download Docker from [Docker's official website](https://www.docker.com/get-started).

**Step 2: Clone the Repository**

```bash
git clone https://github.com/matthewhand/openai-f5-tts
cd openai-f5-tts
```

**Step 3: Prepare a licensed F5-TTS v1 checkpoint (large download)**

The original instructions cloned the whole multi-gigabyte Hugging Face model repository. This application only needs the **v1 Base** checkpoint in a folder matching the voice name. The upstream publisher identifies `F5TTS_v1_Base/model_1250000.safetensors` as the v1 model; the file is approximately **1.35 GB**, is stored using Git LFS/Xet and is licensed **CC-BY-NC-4.0**. This download is optional and **has not been performed or runtime-tested during this review**.

Choose an approved source and confirm you accept the model's license before downloading. The following pinned file and checksum are published by [SWivid on Hugging Face](https://huggingface.co/SWivid/F5-TTS/blob/d6bd6c3c3ec65c0a3ef25a6d3d09658c5e2817fd/F5TTS_v1_Base/model_1250000.safetensors):

- Source revision: `d6bd6c3c3ec65c0a3ef25a6d3d09658c5e2817fd`
- Expected SHA-256: `670900fd14e6c458b95da6e9ed317cdb20dbaf7a1c02ac06a05475a9d32b6a38`
- Destination: `ckpts/Emilia/model_1250000.safetensors`

**Linux (example commands; not executed):**

```bash
mkdir -p ckpts/Emilia ref_audio
curl -fL --retry 3 -o ckpts/Emilia/model_1250000.safetensors \
  https://huggingface.co/SWivid/F5-TTS/resolve/d6bd6c3c3ec65c0a3ef25a6d3d09658c5e2817fd/F5TTS_v1_Base/model_1250000.safetensors
echo "670900fd14e6c458b95da6e9ed317cdb20dbaf7a1c02ac06a05475a9d32b6a38  ckpts/Emilia/model_1250000.safetensors" | sha256sum -c -
curl -fL -o ref_audio/Emilia.wav \
  https://github.com/SWivid/F5-TTS/raw/refs/heads/main/src/f5_tts/infer/examples/basic/basic_ref_en.wav
```

**Windows PowerShell (example commands; not executed):**

```powershell
New-Item -ItemType Directory -Force ckpts/Emilia, ref_audio | Out-Null
curl.exe -fL -o ckpts/Emilia/model_1250000.safetensors "https://huggingface.co/SWivid/F5-TTS/resolve/d6bd6c3c3ec65c0a3ef25a6d3d09658c5e2817fd/F5TTS_v1_Base/model_1250000.safetensors"
(Get-FileHash ckpts/Emilia/model_1250000.safetensors -Algorithm SHA256).Hash
# Compare the output to the SHA-256 above; do not proceed if it differs.
curl.exe -fL -o ref_audio/Emilia.wav "https://github.com/SWivid/F5-TTS/raw/refs/heads/main/src/f5_tts/infer/examples/basic/basic_ref_en.wav"
```

On macOS, use `shasum -a 256 ckpts/Emilia/model_1250000.safetensors` and compare the printed digest with the expected SHA-256 above; `sha256sum` is generally not preinstalled. The *example* reference audio comes from the upstream project and remains subject to its own rights. You may instead supply appropriately licensed/consented audio as `ref_audio/Emilia.wav`. The reference-audio URL above is not pinned; verify source integrity before relying on it in production. Never use untrusted pickle-based `.pt` checkpoints.

**Before running:** `.env.example` contains the nonfunctional placeholder `API_KEY=your_api_key_here`. Replace it with a strong private secret, and set `DEFAULT_VOICE=Emilia` to match the model folder you actually populated. Never commit `.env`. Docker Compose now publishes port 9090 **only on host loopback** (`127.0.0.1:9090`). Do not forward this port externally.

**Step 4: Set Up Environment Variables**

Copy `.env.example` to `.env` and update it as needed:

- **Linux/macOS:**

  ```bash
  cp .env.example .env
  ```

- **Windows (Command Prompt and PowerShell):**

  ```cmd
  copy .env.example .env
  ```

**Key Environment Variables:**

- `API_KEY`: Set this to secure API access.
- `PORT`: The port for the Flask server (default is `9090`).
- `REQUIRE_API_KEY`: Keep `True` (default) to enforce API key authentication. With `True`, supply a non-placeholder `API_KEY`. Setting `False` disables authentication and permits startup without a key **only if the server bind is an explicit numeric loopback IP**: set `HOST=127.0.0.1` or pass `--host 127.0.0.1`. The container normally listens on `0.0.0.0` internally, so **Compose must not start in unauthenticated mode**, even though its published host port is currently loopback. `localhost` hostnames are intentionally rejected when authentication is disabled.
- `DEFAULT_VOICE`: Set explicitly to the exact directory name under `ckpts/` (for this walkthrough: `Emilia`). The current Compose default is also `Emilia`; its checkpoint and `ref_audio/Emilia.wav` must exist.
- `DEFAULT_RESPONSE_FORMAT`: Output audio format (e.g., `mp3`).

**Step 5: Run the Application**

Use Docker Compose to build and start the service:

```bash
docker compose up --build -d
```

*This command may take several minutes to complete, especially the first time as Docker downloads the necessary images.*

**Caution:** Compose now starts the API directly, rather than running a potentially failing inference test first. An authenticated models-list response **does not** establish that actual synthesis works. The Docker profile assumes NVIDIA GPU support and may download the large Whisper model at startup; an audio smoke test needs separate approval.

**Step 6: Access the API**

The API will be available at `http://localhost:9090` by default.

**Step 7: Manage the Service**

- **Stop the service:**

  ```bash
  docker compose down
  ```

- **Restart the service:**

  ```bash
  docker compose up -d
  ```

**Step 8: Confirm Service is Operational**

Use `curl` to verify that the service is running correctly and that the "Emilia" voice model is available:

```bash
curl -H "Authorization: Bearer <YOUR_REAL_API_KEY>" http://127.0.0.1:9090/v1/models
```

You should see a `models` list if initialization succeeds, but a model can appear as loaded without any successful speech request. Confirm actual audio only with an approved local inference smoke test.

---

## Development Setup

If you are developing or contributing to this project, the following method allows for a more flexible setup.

### Prerequisites

- **Python 3.10**: Ensure Python 3.10 is installed on your system.
- **Conda** (recommended) or `venv` for creating an isolated Python environment.

*Note for Windows Users:* If `conda` commands are unavailable, install [Anaconda](https://www.anaconda.com/products/distribution) and use the Anaconda Prompt or Anaconda PowerShell.

### Installation Steps

**Step 1: Clone the Repository**

```bash
git clone https://github.com/matthewhand/openai-f5-tts
cd openai-f5-tts
```

**Step 2: Prepare the same checkpoint and reference audio**

Follow the pinned **F5-TTS v1** model instructions in the [Docker installation section](#using-docker-compose-experimental-local-preview), including license review and SHA-256 verification. The development and Docker paths use the same `ckpts/Emilia/` and `ref_audio/Emilia.wav` layout. Do not clone an entire model repository unnecessarily.

**Step 3: Create a Python Environment**

Set up a Python 3.10 environment using Conda or virtual environments:

- **Using Conda:**

  ```bash
  conda create -n f5-tts python=3.10
  conda activate f5-tts
  ```

  *Note for Windows Users:* If `conda` commands are unavailable, install [Anaconda](https://www.anaconda.com/products/distribution) and use the Anaconda Prompt or Anaconda PowerShell.

- **Using `venv`:**

  ```bash
  python3.10 -m venv f5-tts
  source f5-tts/bin/activate  # On Windows: f5-tts\Scripts\activate
  ```

**Step 4: Install PyTorch and Torchaudio**

Install PyTorch and Torchaudio with CUDA support (adjust for your CUDA version if necessary):

```bash
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

*Note:* Ensure your system has the appropriate CUDA version installed for GPU acceleration. If you do not have CUDA, you can install the CPU-only versions:

```bash
pip install torch==2.3.0+cpu torchaudio==2.3.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

**Step 5: Install Project Dependencies**

Install the required Python packages:

```bash
pip install -r requirements.txt
```

**Step 6: Set Up Environment Variables**

Copy `.env.example` to `.env` and update the variables as needed:

- **Linux/macOS:**

  ```bash
  cp .env.example .env
  ```

- **Windows (Command Prompt/PowerShell):**

  ```cmd
  copy .env.example .env
  ```

**Step 7: Run the Application**

Start the Flask server locally:

```bash
python app/server.py
```

The API will be accessible at `http://localhost:9090`.

---

## API Endpoints

**Busy behavior:** both synthesis endpoints return HTTP `503` and `{"error": "Speech generator busy; try again later"}` if another request already owns the single local generation slot. Non-speech list endpoints are unaffected. This behavior is only process-local.

### `/v1/audio/speech`

Primary route for generating speech from text input. Requires an API key in the request header as a Bearer token.

- **URL**: `/v1/audio/speech`
- **Method**: `POST`
- **Headers**: `Authorization: Bearer <API_KEY>`
- **Data (JSON)**:
  - `input` (string): The text to convert to speech.
  - `voice` (string, optional): Voice model to use (default: "Emilia").
  - `response_format` (string, optional): Output audio format (default: `mp3`).
  - `speed` (float, optional): Speech speed adjustment factor.
  - `ref_audio`: **Not implemented as a per-request override.** Reference audio is resolved from the voice name under `ref_audio/`; omitting this field is recommended.

- **Response**: An audio file in the specified format.

**Example:**

```bash
curl -X POST http://localhost:9090/v1/audio/speech \
     -H "Authorization: Bearer <API_KEY>" \
     -H "Content-Type: application/json" \
     -d '{
           "input": "Hello world",
           "voice": "Emilia",
           "response_format": "mp3",
           "speed": 1.0
         }' > output.mp3
```

### `/v1/models`

Lists available TTS models.

- **URL**: `/v1/models`
- **Method**: `GET`
- **Headers**: `Authorization: Bearer <API_KEY>`
- **Response**: JSON containing the available models.

### `/v1/voices`

Lists available voices, with optional language filtering.

- **URL**: `/v1/voices`
- **Method**: `GET`
- **Headers**: `Authorization: Bearer <API_KEY>`
- **Parameters**:
  - `language` or `locale` (optional): Language filter for voices.
- **Response**: JSON containing the available voices.

### `/v1/text-to-speech/{voice}/with-timestamps`

Optional ElevenLabs-shaped route (not a promise of ElevenLabs interoperability): authenticated POST with `{"text":"Hello"}` (or `input`), and the voice in the URL. The response contains `audio_base64`, `alignment`, `normalized_alignment`, `words` and `alignment_source`. Timing is either `duration_weighted` estimated from text and audio duration, or `unavailable` with null alignment. **No forced aligner is currently connected**, even if WhisperX is installed.

### `/v1/voices/all`

Lists all supported voices, regardless of language.

- **URL**: `/v1/voices/all`
- **Method**: `GET`
- **Headers**: `Authorization: Bearer <API_KEY>`
- **Response**: JSON containing all supported voices.

---

## Adding Your Own Fine-Tuned Checkpoint

Enhance the TTS system by adding your own fine-tuned voice model. Follow these steps:

**Step 1: Create a Directory for Your Voice Model**

- **Linux/macOS:**

  ```bash
  mkdir -p ckpts/MyVoice
  ```

- **Windows (Command Prompt/PowerShell):**

  ```powershell
  mkdir ckpts\MyVoice
  ```

**Step 2: Add the Model File**

Use a trusted **native safetensors export** of your fine-tuning checkpoint named `model_1200000.safetensors` (or `model_1250000.safetensors`). The service no longer discovers `.pt` files and **does not** convert them automatically; obtain the export separately in your trusted training environment. Changing the extension of a `.pt` file is not a conversion. Never load an untrusted pickle checkpoint merely to export it.

- **Linux/macOS:**

  ```bash
  mv path_to_your_checkpoint/model_1200000.safetensors ckpts/MyVoice/
  ```

- **Windows (Command Prompt/PowerShell):**

  ```powershell
  move path_to_your_checkpoint\model_1200000.safetensors ckpts\MyVoice\
  ```

**Step 3: Add Reference Audio**

Add a reference audio file in the `ref_audio/` directory:

- **Linux/macOS:**

  ```bash
  mv path_to_ref_audio/MyVoice.wav ref_audio/
  ```

- **Windows (Command Prompt/PowerShell):**

  ```powershell
  move path_to_ref_audio\MyVoice.wav ref_audio\
  ```

**Step 4: Verify the checkpoint format**

Confirm you have a regular file under `ckpts/MyVoice/` ending in `.safetensors`, not merely a renamed `.pt` file. No automatic `.pt` conversion occurs. Checkpoints must be trusted and licensed; the extension check is **not** a guarantee that an arbitrary model file is safe or functional. The service cannot be called release-ready until actual licensed-model inference is separately approved and verified.

**Step 5: Use the New Voice**

Specify your new voice in API calls:

```bash
curl -X POST http://localhost:9090/v1/audio/speech \
     -H "Authorization: Bearer <API_KEY>" \
     -H "Content-Type: application/json" \
     -d '{
           "input": "Hello world",
           "voice": "MyVoice",
           "response_format": "mp3",
           "speed": 1.0
         }' > output.mp3
```

---

## TODO

- [x] Expose OpenAI-compatible endpoint
- [ ] Independently verify Docker/CUDA compatibility with an approval-gated clean build and licensed model
- [x] Multiple voice models
- [ ] Add expression parsing for nuanced speech
- [x] Document the checkpoint filename discovery rules (see below); independently verify the setup with licensed checkpoints before declaring this complete
- [ ] Enhance error handling and logging

---

## Local offline regression checks

Without downloading models, building containers or sending HTTP requests, run from the repository root:

```bash
python -m unittest discover -s tests -p 'test_request_validation.py' -v
python -m unittest discover -s tests -p 'test_cli.py' -v
python -m unittest discover -s tests -p 'test_source_contract.py' -v
python -m unittest discover -s tests -p 'test_model_cache_failure.py' -v
python -m unittest discover -s tests -p 'test_flask_auth_integration.py' -v
python -m unittest discover -s tests -p 'test_output_limits.py' -v
python -m unittest discover -s tests -p 'test_reference_ffmpeg_timeout.py' -v
```

Run all local low-load synthetic regressions with `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py' -v` (110 tests passed in an isolated source-only workspace on 14 October 2026, including reference-audio and cache symlink/hardlink/path-swap regressions, checkpoint path-trust checks, startup-binding and real-Flask tests; no models, real FFmpeg or inference). These cover pure input validation, mock CLI behavior, generation admission, bounded HTTP audio reads and oversize 503 responses, fake FFmpeg timeouts and reference-cache cleanup, source contracts, **real Flask test-client authentication**, and AST-extracted cache-load/eviction recovery. The Flask integration suite skips when Flask is not already installed; no package installation is needed for the other tests. The startup-binding checks use real Flask imports with inert Gevent and model fixtures; they never open a socket. No actual FFmpeg binary, real model loading/inference, working Docker build, reverse proxy or synthesized audio was validated. No GitHub Actions workflows exist in this repository and this project is not configured to run hosted CI.

The optional CLI launcher is `python f5_tts_app/cli.py --host 127.0.0.1 --port 9090`; it launches the same API server and therefore still needs all models and dependencies installed.

## Support

If you encounter any issues or have questions, feel free to open an issue on the [GitHub repository](https://github.com/matthewhand/openai-f5-tts). Include the API route, error and local environment but **never** API keys, private reference audio or generated text.

---

## Contributing

Contributions are welcome through a focused issue or pull request. There is no published `CONTRIBUTING.md` yet; use this README as the current source of supported behavior.

---

## License

The **wrapper code** is licensed under the [MIT License](LICENSE). **Pretrained F5-TTS model weights are separately licensed CC-BY-NC** by their upstream publisher; commercial use is not granted by this repository's MIT license. Validate the terms of any additional checkpoints and reference audio you supply.

---

Thank you for using **openai-f5-tts**! We hope it helps bring your projects to life with natural and customizable speech synthesis.
