# openai-f5-tts

This project provides a Flask-based API for generating high-quality text-to-speech (TTS) audio using F5-TTS, a flexible and powerful TTS engine. The API supports customizable voices, including the default voice Emilia, and allows for easy integration into applications requiring speech synthesis.

## Setup Instructions

### Prerequisites

- Python 3.10
- `conda` or `venv` (optional) for creating an isolated Python environment

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/matthewhand/openai-f5-tts
   cd openai-f5-tts
   ```

2. **Set Up Environment**

   Follow the F5-TTS setup for compatibility by creating a Python 3.10 environment, then install PyTorch and torchaudio with CUDA support if required.

   ```bash
   # Create a Python 3.10 conda environment
   conda create -n f5-tts python=3.10
   conda activate f5-tts

   # Install PyTorch and torchaudio with CUDA 11.8 support (or your CUDA version)
   pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install Dependencies**

   Install the remaining dependencies from the requirements file.

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**

   Copy `.env.example` to `.env` and update as needed. Ensure that `REF_AUDIO_PATH` points to a reference audio file (detailed in the steps below).

   ```bash
   cp .env.example .env
   ```

   Key environment variables:

   - `API_KEY`: Set this to secure API access.
   - `PORT`: The port for the Flask server (default is 5060).
   - `REQUIRE_API_KEY`: Set to `True` to require API key authentication.
   - `DEFAULT_VOICE`: Default voice name (default is “Emilia,” which points to the default F5-TTS model).
   - `DEFAULT_RESPONSE_FORMAT`: The output audio format (e.g., “mp3”).
   - `DEFAULT_SPEED`: Speech speed adjustment factor.

5. **Download Reference Audio**

   Download the example reference audio file:

   ```bash
   cd ref_audio
   curl -L -o Emilia.wav https://github.com/SWivid/F5-TTS/raw/refs/heads/main/src/f5_tts/infer/examples/basic/basic_ref_en.wav
   cd ..
   ```

6. **Download Default F5-TTS Voice Models**

   Navigate to the `ckpts` directory and download the default F5-TTS models. This step provides base voices, including "Emilia."

   ```bash
   git clone https://huggingface.co/SWivid/F5-TTS
   mv F5-TTS/F5TTS_Base ckpts/Emilia # use move not mv on Windows
   ```

7. **Run the Flask Application**

   After completing setup, start the Flask server:

   ```bash
   python app/server.py
   ```

   The API will be available at `http://localhost:5060` by default.

## API Endpoints

### `/v1/audio/speech`

Primary route for generating speech from text input. Requires an API key in the request header as a Bearer token.

- **URL**: `/v1/audio/speech`
- **Method**: `POST`
- **Headers**: `Authorization: Bearer <API_KEY>`
- **Data (JSON)**:
  - `input` (string): The text to convert to speech.
  - `voice` (string, optional): The voice model to use. Defaults to `Emilia`, referencing the stock F5-TTS model.
  - `response_format` (string, optional): Desired audio format (e.g., `mp3`).
  - `speed` (float, optional): Speed adjustment factor.
  - `ref_audio` (string, optional): Reference audio file path.

- **Response**: Audio file in the requested format.

Example usage:

```bash
curl -X POST http://localhost:5060/v1/audio/speech \
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

### `/v1/voices/all`

Lists all supported voices, regardless of language.

- **URL**: `/v1/voices/all`
- **Method**: `GET`
- **Headers**: `Authorization: Bearer <API_KEY>`
- **Response**: JSON containing all supported voices.

## Adding Your Own Fine-Tuned Checkpoint

After using the official F5-TTS Gradio app to fine-tune a model, you can integrate your checkpoint into this project:

1. **Create a Directory for Your Voice Model**

   In the `ckpts/` directory, create a new folder for your voice (e.g., `Chris`).

   ```bash
   mkdir ckpts/Chris
   ```

2. **Add Your Checkpoint File**

   Place the `model_1200000.pt` file from your fine-tuning into the new folder:

   ```bash
   mv path_to_your_checkpoint/model_1200000.pt ckpts/Chris/
   ```

3. **Provide Reference Audio**

   Add a reference audio file for your voice in the `ref_audio/` directory. Ensure the file name matches the voice directory name:

   ```bash
   mv path_to_ref_audio/Chris.wav ref_audio/
   ```

4. **First Run**

   On the first run, the application will automatically convert the `.pt` file into a `.safetensors` file for efficiency and security. Subsequent runs will use the `.safetensors` file.

5. **Use Your Voice**

   To generate speech with your custom model, specify the voice in the API call:

   ```bash
   curl -X POST http://localhost:5060/v1/audio/speech \
        -H "Authorization: Bearer <API_KEY>" \
        -H "Content-Type: application/json" \
        -d '{
              "input": "Hello world",
              "voice": "Chris",
              "response_format": "mp3",
              "speed": 1.0
            }' > output.mp3
   ```

## TODO

- [x] Expose OpenAI-compatible endpoint
- [x] Fix Docker + CUDA compatibility
- [x] Multiple voice models
- [ ] Add expression parsing for nuanced speech
- [ ] Document usage for fine-tuned models
- [ ] Enhance error handling and logging
