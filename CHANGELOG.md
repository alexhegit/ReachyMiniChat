# Changelog

All notable changes to this repository will be documented in this file.

## [Unreleased]
- Add `emo_v8_openclaw.py`: OpenClaw integration (replaces Ollama with OpenClaw API).
  - OpenAI-compatible API integration with proper authentication
  - Support for streaming responses and emotion analysis
  - Includes ASR and TTS capabilities
- Remove `emo_v8.py`: Legacy file replaced by OpenClaw integration
- Add `emo_v7.py`: ASR (faster-whisper CPU) → Ollama → Edge-TTS + emotion-driven actions.
  - Push-to-talk ASR mode (4s recordings by default).
  - Integration with `EmotionControllerV6` for emotion analysis and synchronized actions.
  - Docs: `EMO_V7_README.md` and updated `EMO_README.md`/`README.md` links.

## [v6]
- Continuous synchronized actions and Edge-TTS cartoon voice integration (see `emo_v6.py`).

## [v5]
- Edge-TTS integration and WAV playback improvements.


