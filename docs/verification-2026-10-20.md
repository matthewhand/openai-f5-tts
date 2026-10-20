# Docker Desktop verification: 20 October 2026

This verifies the independently developed openai-f5-tts API wrapper using upstream
F5-TTS models. It is not a claim of affiliation with OpenAI or the upstream authors.

## Environment and scope

The application source at commit `ce4e90e` was rebuilt and run on Windows with
Docker Desktop's Linux backend and an NVIDIA RTX 3080 Ti (12 GB). The service log
selected `cuda:0`. Its operator-selected host port was 9091; the repository's
default host port is 9090.

Runtime: Python 3.10.22, PyTorch 2.14.1+cu130, F5-TTS 1.1.22 and Transformers 4.57.6.
The installed upstream F5-TTS Git revision was
`2f866649b779881af8bd644e76b9a6d95053da29`.

The application build reused its cached dependency layer. This is verification of
the existing local deployment with current wrapper source, rather than a cold
dependency installation. Upstream dependencies are still unpinned in requirements.txt.
Existing checkpoints and reference audio were used; none were uploaded to GitHub.

## Real model results

| Request | Result |
|---|---|
| Scarlett v0, non-streaming | Non-silent WAV, 24 kHz mono; requested words transcribed correctly |
| Emilia v1, non-streaming | Non-silent WAV, 24 kHz mono; requested words transcribed correctly |
| Emilia, MP3 | Successfully decoded as non-silent 24 kHz mono MP3 |
| Emilia, two-sentence streaming | Incremental 16 kHz mono int16 PCM; both sentences transcribed correctly |
| Scarlett, two-sentence streaming | Incremental 16 kHz mono int16 PCM; both sentences transcribed correctly |
| Configured default voice, omitted from request | Successful Scarlett WAV |
| Timestamp route | Audio plus duration-weighted estimated alignment |
| Wrong API credentials | HTTP 401 on all seven routes |
| Empty text, unknown voice, invalid speed | HTTP 400 on ordinary synthesis route |

For both non-streaming voice samples, Whisper recovered the words:
"The quick brown fox jumps over the lazy dog."

The Emilia stream contained:
"This is the first streamed sentence. This is the second streamed sentence."

The Scarlett stream contained:
"The first voice can stream sentences. The second sentence is delivered next."

The stream tests received audio before the entire request completed. Scarlett's
first audio arrived after 2.813 seconds, with the response complete after 5.360
seconds for 5.035 seconds of audio. Emilia's first tested stream included initial
processing overhead: first audio after 14.812 seconds and completion after 16.531
seconds for 7.445 seconds of audio. These are individual samples, not latency guarantees.

The Emilia checkpoint matched the README's published SHA-256 pin. Both checkpoint
configuration paths (v0 and v1) produced intelligible speech according to independent
transcription of the generated outputs. Subjective listening quality was not assessed.

## Regressions

164 tests and 136 subtests passed in a disposable container using the rebuilt image
and the repository's committed default Compose configuration. One local-only port
customization initially failed the test that asserts the default port; the default
file was mounted only in the test container for the passing run. The operator's
live port configuration was preserved.

## Multiple voices and remaining limits

Both installed voices worked through both synthesis modes, in sequential requests.
`MODEL_LOAD_LIMIT=1` keeps one voice checkpoint loaded and swaps it when the selected
voice changes. More installed voices do not require more simultaneous loaded models.

The streaming implementation currently does not share the ordinary generation
admission gate. Simultaneous streaming, overlapping voice switches and cancellation
during native inference remain unverified and must not be treated as safe concurrent
operation. Raising MODEL_LOAD_LIMIT can retain more models if memory permits; it
does not solve request concurrency.

CPU-only operation, sustained load, every output format and cold dependency
installation were not established by these checks. Timestamp alignment is estimated,
not forced alignment. Upstream pretrained F5-TTS weights remain CC-BY-NC; rights to
custom checkpoints and reference audio depend on their provenance.
