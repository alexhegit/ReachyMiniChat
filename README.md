# Reachy Mini — Ollama Chat + Emotion/Dance Demo


"Don't have physical hardware? You can still create your own virtual robot on your desk. This represents a straightforward sim2real practice leveraging MuJoCo and AI tools like Faster Whisper, Ollama, and espeak/edgeTTS. While edgeTTS relies on cloud APIs, espeak enables fully offline operation. I developed this on the AMD Strix Halo platform and tested it on an AMD Radeon GPU with Ubuntu. Although untested on other systems, the architecture should facilitate easy porting to macOS and Windows."


![Demo](./assets/ReachyMiniChat.png)

## Short summary
- This repository contains demo apps and controllers for the Reachy Mini simulator and small robot, focused on emotion-driven and dance actions triggered from language model outputs (Ollama). It includes several experimental versions (`emo_v1` → `emo_v7`) that explore recorded-move playback, streaming-triggered motions, and TTS integration.

What you'll find
- `emo_v1.py` — Baseline high-intensity emotion controller and examples.
- `emo_v2.py` — RecordedMoves categorization and selection.
- `emo_v3.py` — Streaming LM responses triggering actions early.
- `emo_v4.py` — Offline- focused TTS (espeak) with lip-sync hooks.
- `emo_v5.py` — Edge-TTS integration with WAV save/read/play flow (multi-language support).
- `emo_v6.py` — Continuous synchronized actions with cartoon voices and multi-modal expressions.
- `emo_v7.py` — ASR → LLM → TTS demo (see EMO_V7_README.md)
- `emo_v8_openclaw.py` — OpenClaw integration (replaces Ollama with OpenClaw API)

## Installation prerequisites (Linux / Debian-family)
1. System packages

```bash
sudo apt update
sudo apt install -y python3 python3-dev python3-venv python3-pip build-essential \
                    espeak ffmpeg libsndfile1 portaudio19-dev libcairo2-dev 
```

**Notes:**
- `espeak` is required for the offline TTS flow used by `emo_v4.py`.
- `libsndfile1` and `portaudio` are required for `soundfile` and `sounddevice` (used when playing WAVs).
- `ffmpeg` is optional but useful if you need to convert audio formats or debug audio files.

2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Ollama / Reachy Mini SDK
- This repo uses Ollama and Reachy Mini SDK for LLM and action responses in demos. Please follow those tools' own install instructions.

Install reachy-mini SDK with Mujoco support:
```bash
pip install "reachy-mini[mujoco]"
```

Install Ollama from https://ollama.com/download. Then install it and pull Qwen3:0.6B which is the LLM we used in this repo.

## Run it

1. Start the Reachy Mini simulation in terminal 1:

```bash
reachy-mini-daemon --sim
```


2. Quick test commands (terminal 2)

```bash
# Run the action tests (plays recorded moves + emotions)
python ./utils/test_actions.py

# Test TTS in emo_v5 (Edge-TTS path) — the script includes a --test-tts flag in emo_v5
python emo_v5.py --test-tts

# Test espeak offline TTS in emo_v4
python emo_v4.py --test-tts
```

## Project notes and troubleshooting
- If you hear noisy or distorted audio, ensure `soundfile` and `sounddevice` are installed in the active venv, and that the system `libsndfile` and PortAudio development packages are present.
- `emo_v5.py` writes Edge-TTS output to WAV and plays it back using the file's sample rate to avoid playback artifacts.
- `emo_v4.py` uses `espeak --stdout` as the primary offline TTS backend; ensure `espeak` is installed.

emo_v7 (ASR → LLM → TTS)
---------------------------------
- `emo_v8_openclaw.py` integrates with OpenClaw API instead of Ollama, using OpenAI-compatible endpoints with proper authentication. See [EMO_V8_OPENCLAW.md](EMO_V8_OPENCLAW.md) for usage, requirements, and setup instructions.
- `emo_v7.py` adds a microphone-first pipeline using `faster-whisper` (CPU) for ASR, then forwards the transcription to Ollama and uses the existing emotion controller + Edge-TTS for speech and actions.
- See [EMO_V7_README.md](EMO_V7_README.md) for usage, requirements, and notes about model choices and VAD improvements.

Changelog
---------------------------------
- See [CHANGELOG.md](CHANGELOG.md) for recent changes and highlights across `emo_v*` versions.
