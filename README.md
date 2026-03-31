# Reachy Mini — Ollama Chat + Emotion/Dance Demo


"Don't have physical hardware? You can still create your own virtual robot on your desk. This represents a straightforward sim-to-real practice leveraging MuJoCo and AI tools like Faster Whisper, Ollama, and eSpeak/Edge-TTS. While Edge-TTS relies on cloud APIs, eSpeak enables fully offline operation. I developed this on the AMD Strix Halo platform and tested it on an AMD Radeon GPU with Ubuntu. Although untested on other systems, the architecture should facilitate easy porting to macOS and Windows."


![Demo](./assets/ReachyMiniChat.png)

## Short summary
- This repository contains demo apps and controllers for the Reachy Mini simulator and small robot, focused on emotion-driven and dance actions triggered from language model outputs (Ollama). It includes multiple experimental versions of increasing capability (`emo_v1` → `emo_v8`) that explore recorded-move playback, streaming-triggered motions, and TTS integration.

For a quick summary of each emo_v*.py iteration:
- `emo_v1.py` — Baseline text chat, high-amplitude emotion controller, and examples.
- `emo_v2.py` — Swaps hardcoded movements for the RecordedMoves library (richer, more natural motion vocabulary).
- `emo_v3.py` — Streams LM responses which triggers robot actions earlier instead of waiting for full response.
- `emo_v4.py` — Adds an offline-focused TTS (eSpeak) voice output with basic lip-sync hooks.
- `emo_v5.py` — Upgrades TTS to Edge-TTS integration with WAV save/read/play flow (multi-language support).
- `emo_v6.py` — Continuous synchronized actions throughout speech — with cartoon voices and multi-modal expressions.
- `emo_v7.py` — Adds voice intput — ASR → LLM → TTS demo (see EMO_V7_README.md).
- `emo_v8.py` — Uses offline Piper-TTS version (ASR/text chat + Ollama + Piper).

## Installation prerequisites (Linux / Debian-family)

This project is developed on an AMD Ryzen™ AI Max+ 395 running Ubuntu 24.04. We recommend this hardware for deploying the application, as it serves as an excellent companion to the Reachy Mini Desktop Robot. The integrated GPU and CPU provide the necessary performance to run the full pipeline 100% offline.

So you may follow the [AMD ROCm Documentation](
https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html) to install Ryzen Software for Linux with ROCm.

Then go to setup the environment for this application. (*Note: the required system packages should already be installed for you; if you would like to view them, you can do so [here](./setup_deps.sh)*)

1. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Reachy Mini SDK

This repo uses the Reachy Mini SDK to program, control, and simulate robot actions in the demo. Please follow the tool's own installation instructions.

- Install reachy-mini SDK with Mujoco support:
```bash
pip install "reachy-mini[mujoco]"
```

3. Ollama

Ollama is an open-source tool that we'll use to run our models locally. (*Note: Ollama should already be installed for you, but here is the download link for your reference: https://ollama.com/download*)

- Pull the Qwen3:0.6B model which is the LLM we used in this repo.
```bash
ollama pull qwen3:0.6b
ollama serve
```

## Run it

1. Start the Reachy Mini simulation in terminal 1:
- Use `export PYGLFW_LIBRARY_VARIANT=x11` if the GUI launch fails on Wayland, which is the default backend of Ubuntu 24.04+.
```bash
reachy-mini-daemon --sim
```

2. Quick test commands in terminal 2:

```bash
# Run the action tests (plays recorded moves + emotions)
python ./utils/test_actions.py

# Test TTS in emo_v5 (Edge-TTS path) — the script includes a --test-tts flag in emo_v5
python emo_v5.py --test-tts

# Test eSpeak offline TTS in emo_v4
python emo_v4.py --test-tts
```

## Project notes and troubleshooting
- If you hear noisy or distorted audio, ensure `soundfile` and `sounddevice` are installed in the active venv, and that the system `libsndfile` and PortAudio development packages are present.
- `emo_v5.py` writes Edge-TTS output to WAV and plays it back using the file's sample rate to avoid playback artifacts.
- `emo_v4.py` uses `espeak --stdout` as the primary offline TTS backend; ensure eSpeak is installed.

## emo_v7 (ASR → LLM → TTS)
- `emo_v7.py` adds a microphone-first pipeline using `faster-whisper` (CPU) for ASR, then forwards the transcription to Ollama and uses the existing emotion controller + Edge-TTS for speech and actions.
- See [EMO_V7_README.md](EMO_V7_README.md) for usage, requirements, and notes about model choices and VAD improvements.
- New CLI flag: `--gentle` — enables gentle_mode which restricts selected recorded moves to a curated gentle set and adjusts motion durations for subtler actions. Example:

```bash
python emo_v7.py --asr --gentle
```

## emo_v8 (Offline Piper-TTS)
- `emo_v8.py` replaces Edge-TTS with Piper-TTS for fully offline speech synthesis, while keeping Ollama chat and emotion/action flow.
- New dependency is already included in `requirements.txt`:
  - `piper-tts>=1.4.0`
- `emo_v8.py` also supports `--gentle` (same behavior as emo_v7/emo_v6) and accepts `--piper-model` and `--piper-config` to point to local voice models. Example:

```bash
python emo_v8.py --model qwen3:0.6b --piper-model models/zh_CN-huayan-medium.onnx --gentle
```
To download the models from the Hugging Face hub:
- `csukuangfj/vits-piper-zh_CN-huayan-medium`
- `csukuangfj/vits-piper-en_US-lessac-medium`

If you'd like to download and experiment with more piper voice models, you can find further `.onnx` and matching `.onnx.json` voice files from:
- Piper release page: `https://github.com/rhasspy/piper/releases/tag/v0.0.2`
- Voice files repo: `https://huggingface.co/rhasspy/piper-voices`

Place your chosen files under `models/` (or any path you pass to `--piper-model`).

Example usage (with a qwen3.5:0.8b model instead):
```bash
# Text chat mode + english (default)
python emo_v8.py --model qwen3.5:0.8b --piper-model models/en_US-lessac-medium.onnx

# ASR mode + Chinese 
python emo_v8.py --asr --model qwen3.5:0.8b --piper-model models/zh_CN-huayan-medium.onnx

# ASR + gentle action + Chinese
python emo_v8.py --piper-model ./models/zh_CN-huayan-medium.onnx --gentle --model qwen3.5:0.8b

# Optional: explicit Piper config/speaker
python emo_v8.py --piper-model models/en_US-lessac-medium.onnx --piper-config models/en_US-lessac-medium.onnx.json --speaker 0
```

## Version History
- See [EMO_README.md](EMO_README.md) for version details and changelog across `emo_v*` versions.
