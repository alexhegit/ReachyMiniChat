# EMO README — Reachy Mini Emotion Versions (Merged)

This document consolidates per-version summaries and a full version-comparison table for `emo_v1.py` → `emo_v7.py`.

---

## Full Version Comparison Table

| Feature | emo_v1 | emo_v2 | emo_v3 | emo_v4 | emo_v5 | emo_v6 | emo_v7 | emo_v8_openclaw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Action Source | custom | recorded moves | recorded moves | recorded moves | recorded moves | recorded moves | recorded moves | recorded moves |
| Emotion Types | 4 basic | 4 enhanced | 4 enhanced | 4 enhanced | 4 enhanced | 4 enhanced | 4 enhanced | 4 enhanced |
| Action Timing | after text | after text | during text | during speech | during speech | continuous speech | ASR → during speech | ASR → during speech |
| TTS Engine | none | none | none | multi-backend (local) | Edge-TTS (cloud) | Edge-TTS cartoon voices | Edge-TTS cartoon voices | Edge-TTS cartoon voices |
| Lip-sync | no | no | no | generic | antenna/eye precise | multi-modal synchronized | multi-modal synchronized | multi-modal synchronized |
| Voice Quality | N/A | N/A | N/A | local | neural cloud | cute cartoon + parameters | cute cartoon + parameters | cute cartoon + parameters |
| Threading | no | no | yes | yes | yes | advanced multi-thread | advanced multi-thread | advanced multi-thread |
| Emoji Support | no | yes | yes | yes | yes | yes | yes | yes |
| Eye Blinking | no | no | no | no | no | synchronized | synchronized | synchronized |
| Body Yaw | no | no | no | no | no | synchronized | synchronized | synchronized |
| Action Variety | 1 per emotion | 1 per emotion | 1 per emotion | 1 per emotion | 1 per emotion | 4-5 sequences per emotion | 4-5 sequences per emotion | 4-5 sequences per emotion |

---

---

## emo_v1 — High-Intensity Emotion Engine

Summary
- Purpose: Baseline high-intensity emotion controller and interactive chat app.
- Key features: high-intensity defaults, large-amplitude motion mappings, simple keyword-based emotion classification, quick test harness.

Code snapshot
```python
# emo_v1: basic analysis
text_lower = text.lower()
if any(w in text_lower for w in ["跳舞", "舞蹈"]):
    emotion = "activity"
controller.perform_high_amplitude_action(emotion)
```

---

## emo_v2 — Recorded Moves Integration

Summary
- Purpose: Use `RecordedMoves` for a richer motion vocabulary and categorize moves by emotion.

Code snapshot
```python
# emo_v2 categorize recorded moves
all_moves = self.recorded_moves.list_moves()
for move_name in all_moves:
    move = self.recorded_moves.get(move_name)
    desc = move.description.lower() if move.description else ""
    # assign best_match
```

---

## emo_v3 — Parallel Actions During Streaming

Summary
- Purpose: Trigger actions early while Ollama streams the response for better responsiveness.

Code snapshot
```python
# emo_v3 pseudo
for line in response.iter_lines():
    if 'response' in chunk:
        if should_trigger_now(partial):
            start_background_move()
```

---

## emo_v4 — Reliable Offline TTS (espeak) + Lip-sync

Summary
- Purpose: Add deterministic offline TTS using `espeak` and add lip-sync (antenna/eye animation).

Code snapshot (espeak synth)
```python
cmd = ['espeak', '--stdout', text]
result = subprocess.run(cmd, capture_output=True, check=True)
with open(output_file, 'wb') as f:
    f.write(result.stdout)
```

---

## emo_v5 — High-Quality Edge-TTS with Correct Playback

Summary
- Purpose: Use Edge-TTS to create higher-quality, multilingual speech. Save to WAV, read with `soundfile` and play using `sounddevice` at the file's sample rate to avoid noise.

Code snapshot
```python
# emo_v5
await edge_tts.Communicate(text, voice).save(tmp_path)
data, sr = sf.read(tmp_path, dtype='float32')
sd.play(data, samplerate=sr)
sd.wait()
```

---

## emo_v6 — Continuous Synchronized Actions + Cute Cartoon Voices + Integrated Testing

Summary
- Purpose: Major enhancement with continuous emotional actions throughout speech, synchronized eye blinking + body yaw + head poses, and cute cartoon voices. Features 4-5 action sequences per emotion for maximum expressiveness. Includes integrated testing tools with command-line options for comprehensive validation.

Key Features:
- **Integrated Testing**: `--test-actions` and `--test-tts` command-line options
- **Merged Functionality**: test_combined_actions.py functionality integrated into main script
- **Enhanced Documentation**: Detailed action names in test logs
- **Removed Broken Features**: --test-compat option removed due to compatibility issues

Code snapshot
```python
# emo_v6 combined actions with integrated testing
def _combined_nod_blink(self):
    # Eye blink + head nod synchronization
    if hasattr(self.reachy, 'head'):
        self.reachy.head.r_eye.goal_position = 0.1
    self.reachy.goto_target(head=create_head_pose(pitch=20), duration=0.25)
    # ... full synchronization sequence

# Integrated testing with descriptive action names
def test_combined_actions(self):
    action_name = action.__name__.replace('_combined_', '').replace('_', ' ').title()
    print(f"  {i}. Performing {action_name}...")
```

---

## emo_v7 — ASR → LLM → TTS (faster-whisper CPU)

Summary
- Purpose: Add a microphone-first pipeline so users can speak directly to Reachy Mini: ASR (faster-whisper on CPU) → Ollama LLM → Edge-TTS + emotion-driven actions.

What you'll find
- Push-to-talk ASR mode implemented with `faster-whisper` (CPU) in `emo_v7.py`.
- Integration with `EmotionControllerV6` so transcribed text is analyzed for emotion and triggers recorded moves / lip-sync during TTS.
- Simple 4s push-to-talk recording by default; documented suggestions for VAD/whisper.cpp fallbacks.

Requirements
- `faster-whisper`, `sounddevice`, and `soundfile` installed in your environment for ASR recording and transcription.

Quick test
```bash
# ASR push-to-talk mode
python emo_v7.py --asr

# Text chat mode
python emo_v7.py
```

Notes
- The ASR mode uses CPU `faster-whisper` by default (`model='small'` recommended). Replace with `whisper.cpp` or VOSK for different latency/accuracy tradeoffs.
- Consider adding VAD (`webrtcvad`) later to automatically detect end-of-speech instead of fixed-length recording.


## Migration & CLI notes
- Use `controller.execute_emotion_move()` to move from v1→v2 style.
- `emo_v3` introduces streaming hooks; prefer `_get_ollama_response_parallel`.
- `emo_v4` and `emo_v5` add TTS: use `--test-tts` to validate.
- `emo_v6` integrates testing tools: use `--test-actions` for combined action testing, `--test-tts` for Edge-TTS validation.
- `emo_v6` removed `--test-compat` option due to compatibility issues.

---

## Tests
- `python test_actions.py` now validates both dances and emotions recorded-move libraries (limited to a small set by default).
- `python emo_v4.py --test-tts` validates local `espeak` integration.
- `python emo_v5.py --test-tts` validates Edge-TTS (requires network).
- `python emo_v6.py --test-actions` validates v6 synchronized eye blinking + body yaw + head + antennas (integrated testing).
- `python emo_v6.py --test-tts` validates Edge-TTS with emotion analysis and cartoon voices.
- `python utils/test_edge_tts_voices.py` discovers and tests cute voices from Edge-TTS library.

---

## emo_v8_openclaw — OpenClaw API Integration

Summary
- Purpose: Replace Ollama with OpenClaw API integration using OpenAI-compatible endpoints.
- Key features: Proper authentication with Bearer token, OpenClaw-specific headers, seamless integration with existing emotion controller and ASR/TTS capabilities.

Code snapshot
```python
# OpenClaw API request with proper authentication
headers = {
    "Authorization": f"Bearer {self.openclaw_token}",
    "Content-Type": "application/json",
    "x-openclaw-agent-id": "main"
}
payload = {
    "model": "openclaw",  # OpenClaw-specific model name
    "messages": [...],
    "stream": True
}
```

Requirements
- Valid OpenClaw API token
- OpenClaw Gateway running with HTTP API enabled

Quick test
```bash
# Text chat mode with OpenClaw
python emo_v8_openclaw.py --token "your_token_here"

# ASR voice mode with OpenClaw
python emo_v8_openclaw.py --token "your_token_here" --asr
```

Notes
- Uses "openclaw" as the model name instead of specific LLM identifiers
- Requires the x-openclaw-agent-id header for proper routing
- Maintains all existing features from emo_v7 (ASR, TTS, emotion analysis)
