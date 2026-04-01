#!/usr/bin/env python3
"""emo_v9.py - Reachy Mini Chat v9 (Development)

Incremental improvements over v8:
- Step 1: Fix EmotionControllerV71 inheritance (avoid EdgeTTSEngine creation)
- Step 2: (Optional) Add conversation history/context
- Step 3: (Optional) Add performance timing statistics

Usage:
  python emo_v9.py --piper-model models/en_US-lessac-high.onnx --asr
  python emo_v9.py --debug  # Show detailed logs

Development workflow:
  1. Make small change
  2. Test thoroughly  
  3. Commit
  4. Next feature
"""

import os
import sys
import time
import json
import wave
import tempfile
import asyncio
import argparse
import threading
import subprocess
import select
import termios
import tty
import numpy as np
import soundfile as sf
import sounddevice as sd
import aiohttp
from typing import Optional, Tuple, Dict, List
from collections import deque
from contextlib import suppress

# Import from existing modules
# We need to ensure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emo_v6 import EmotionControllerV6, LipSyncControllerV5
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.recorded_move import RecordedMoves

# Optional faster-whisper ASR engine
try:
    from utils.asr import FasterWhisperASREngine
except Exception:
    try:
        from .utils.asr import FasterWhisperASREngine
    except Exception:
        FasterWhisperASREngine = None

class PiperTTSEngine:
    """Piper-TTS engine wrapper for offline speech synthesis."""
    
    def __init__(self, model_path: str, config_path: str = None, speaker_id: int = 0, debug: bool = False):
        self.debug = debug
        self.model_path = model_path
        self.config_path = config_path
        self.speaker_id = speaker_id
        self.voice = None
        self._stop_requested = threading.Event()
        
        try:
            from piper import PiperVoice, PiperConfig
            # Import SynthesisConfig if available, else use default dict
            try:
                from piper import SynthesisConfig
                self.SynthesisConfig = SynthesisConfig
            except ImportError:
                self.SynthesisConfig = None
            
            import onnxruntime
            self.PiperVoice = PiperVoice
            self.PiperConfig = PiperConfig
            self.onnxruntime = onnxruntime
        except ImportError:
            print("❌ piper-tts not installed. Install with: pip install piper-tts")
            return

        if not os.path.exists(model_path):
            print(f"❌ Piper model not found at: {model_path}")
            
            # Try to find any onnx model in models/ or current directory
            print("🔍 Searching for available models...")
            found_models = []
            for search_dir in ['.', 'models']:
                if os.path.exists(search_dir):
                    for f in os.listdir(search_dir):
                        if f.endswith('.onnx'):
                            found_models.append(os.path.join(search_dir, f))
            
            if found_models:
                print(f"💡 Found available models:")
                for m in found_models:
                    print(f"   --piper-model {m}")
                print(f"\nExample: python emo_v8.py --piper-model {found_models[0]}")
            else:
                print("⚠️ No .onnx models found. Please download one from https://github.com/rhasspy/piper/releases/tag/v0.0.2")
                
            self.voice = None
            return

        try:
            # If config path not provided, assume .json with same name as .onnx
            if not config_path:
                potential_config = model_path + ".json"
                if os.path.exists(potential_config):
                    self.config_path = potential_config
            
            print(f"🎙️ Loading Piper model: {model_path}")
            
            # Manually load config to fix legacy phoneme_type issue
            with open(self.config_path or (model_path + ".json"), 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # FIX: Replace legacy "PhonemeType.ESPEAK" string with "espeak"
            if config_dict.get('phoneme_type') == 'PhonemeType.ESPEAK':
                print("🔧 Fixing legacy phoneme_type in config...")
                config_dict['phoneme_type'] = 'espeak'
                
            # Create config object
            config = self.PiperConfig.from_dict(config_dict)
            
            # Create ONNX session
            session = self.onnxruntime.InferenceSession(
                str(model_path),
                sess_options=self.onnxruntime.SessionOptions(),
                providers=["CPUExecutionProvider"]
            )
            
            # Initialize voice manually
            self.voice = self.PiperVoice(session=session, config=config)
                
            print(f"✅ Piper TTS initialized")
            
        except Exception as e:
            print(f"❌ Failed to load Piper model: {e}")
            self.voice = None

    def speak_with_emotion(self, text: str, emotion: str = 'neutral'):
        """Speak text using Piper (blocking)."""
        if not text.strip():
            return
            
        if not self.voice:
            print(f"⚠️ Piper voice not loaded. Skipping speech: '{text[:20]}...'")
            return

        try:
            self._stop_requested.clear()

            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name

            # Synthesize to file
            with wave.open(tmp_path, "wb") as wav_file:
                # Use synthesize_wav which handles wave header automatically
                syn_config = None
                if self.SynthesisConfig and self.speaker_id is not None:
                    syn_config = self.SynthesisConfig(speaker_id=self.speaker_id)
                
                self.voice.synthesize_wav(text, wav_file, syn_config=syn_config)

            # Read and play
            data, sr = sf.read(tmp_path, dtype='float32')
            if data.size > 0:
                sd.play(data, samplerate=sr)
                while True:
                    if self._stop_requested.is_set():
                        sd.stop()
                        break

                    try:
                        stream = sd.get_stream()
                    except Exception:
                        stream = None

                    if stream is None or not stream.active:
                        break

                    time.sleep(0.05)
            
            # Cleanup
            try:
                os.remove(tmp_path)
            except:
                pass
                
        except Exception as e:
            print(f"⚠️ Piper TTS error: {e}")

    async def speak_with_emotion_async(self, text: str, emotion: str = 'neutral'):
        """Async version of speak_with_emotion (runs in thread)."""
        # Piper synthesis is CPU bound, so run in a separate thread
        await asyncio.to_thread(self.speak_with_emotion, text, emotion)

    def stop(self):
        """Stop any in-progress audio playback."""
        self._stop_requested.set()
        try:
            sd.stop()
        except Exception:
            pass


class ConversationHistory:
    """Manages conversation history for context-aware responses."""
    
    def __init__(self, max_rounds: int = 5):
        """
        Initialize conversation history.
        
        Args:
            max_rounds: Maximum number of conversation rounds to keep (default: 5)
        """
        self.max_rounds = max_rounds
        self.history: deque = deque(maxlen=max_rounds * 2)  # Each round has user + assistant
        self.enabled = True
    
    def add_user_message(self, message: str):
        """Add a user message to history."""
        if self.enabled and message.strip():
            self.history.append({"role": "user", "content": message.strip()})
    
    def add_assistant_message(self, message: str):
        """Add an assistant message to history."""
        if self.enabled and message.strip():
            self.history.append({"role": "assistant", "content": message.strip()})
    
    def get_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """
        Get all messages formatted for Ollama API.
        
        Args:
            include_system: Whether to include system prompt
            
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages = []
        
        if include_system:
            messages.append({
                "role": "system",
                "content": "You are a cute desktop robot assistant. Respond with enthusiasm and warmth. Remember the user's name and preferences from the conversation."
            })
        
        messages.extend(list(self.history))
        return messages
    
    def clear(self):
        """Clear all conversation history."""
        self.history.clear()
        print("🗑️  Conversation history cleared")
    
    def get_summary(self) -> str:
        """Get a summary of current history."""
        rounds = len(self.history) // 2
        return f"History: {rounds} rounds (max {self.max_rounds})"


class EmotionControllerV71(EmotionControllerV6):
    """Emotion controller using Piper-TTS instead of Edge-TTS."""
    
    def __init__(self, reachy: ReachyMini, piper_model: str, piper_config: str = None, 
                 speaker_id: int = 0, debug: bool = False, gentle_mode: bool = False):
        # Step 1 Fix: Skip parent __init__ to avoid creating EdgeTTSEngine
        # Instead, directly initialize only what we need
        self.reachy = reachy
        self.debug = debug
        self.gentle_mode = gentle_mode
        self.is_speaking_action = False
        
        # Use Piper TTS directly (no Edge-TTS)
        self.tts_engine = PiperTTSEngine(piper_model, piper_config, speaker_id, debug)
        self.lip_sync = LipSyncControllerV5(reachy, debug=self.debug)

        # Load both libraries for richer motions
        self.emotions_lib = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        self.dances_lib = RecordedMoves("pollen-robotics/reachy-mini-dances-library")

        self._categorize_recorded_moves()

        self.simple_actions = {
            'nod': self._simple_nod,
            'shake': self._simple_shake,
            'look_curious': self._simple_look_curious,
            'look_sad': self._simple_look_sad,
            'excited_wiggle': self._simple_excited_wiggle,
            'thoughtful_tilt': self._simple_thoughtful_tilt,
        }

    def interrupt_speech(self):
        """Stop current TTS playback and wind down animation threads."""
        self.is_speaking_action = False
        self.lip_sync.stop_lip_sync()
        self.tts_engine.stop()


class ChatAppWithPiper:
    def __init__(self, 
                 model: str = "qwen3:0.6b", 
                 ollama_url: str = "http://localhost:11434", 
                 piper_model: str = "en_US-libritts_r-medium.onnx",
                 piper_config: str = None,
                 speaker_id: int = 0,
                 debug: bool = False, 
                 use_asr: bool = False,
                 gentle: bool = False,
                 history_size: int = 5,
                 enable_history: bool = True,
                 asr_model: str = "small",
                 vad_silence: float = 0.8,
                 vad_aggressive: int = 1,
                 use_vad: bool = True):
        self.model = model
        self.ollama_url = ollama_url
        self.debug = debug
        self.use_asr = use_asr
        self.gentle = gentle
        self.piper_model = piper_model
        self.piper_config = piper_config
        self.speaker_id = speaker_id
        self.asr_model = asr_model  # Step 4 B: ASR model selection
        self.vad_silence = vad_silence  # VAD silence threshold (default 0.8s, increase if cutting off)
        self.vad_aggressive = vad_aggressive  # VAD aggressiveness 0-3 (1=gentle, 3=strict)
        self.use_vad = use_vad  # Whether to use VAD or fixed-duration recording
        
        self.controller: Optional[EmotionControllerV71] = None
        self.asr_engine = None
        
        # Step 2: Conversation history
        self.history = ConversationHistory(max_rounds=history_size)
        self.history.enabled = enable_history

    def _wait_for_ctrl_d(self, stop_event: threading.Event) -> bool:
        """Watch stdin for Ctrl-D while allowing Ctrl-C to keep its default behavior."""
        if not sys.stdin.isatty():
            return False

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                ready, _, _ = select.select([fd], [], [], 0.1)
                if not ready:
                    continue

                char = os.read(fd, 1)
                if char == b"\x04":
                    return True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return False

    async def _run_speech_with_interrupt(
        self,
        response: str,
        emotion: str,
        intensity: str,
        emotion_level: float,
    ) -> bool:
        """Run TTS/animation and allow Ctrl-D to skip to the next recording."""
        if self.controller is None:
            return False

        speech_task = asyncio.create_task(
            self.controller.speak_with_expression_parallel(
                response, emotion, intensity, emotion_level
            )
        )

        stop_event = threading.Event()
        interrupt_task = None

        if self.use_asr and sys.stdin.isatty():
            interrupt_task = asyncio.create_task(
                asyncio.to_thread(self._wait_for_ctrl_d, stop_event)
            )

        try:
            if interrupt_task is None:
                await speech_task
                return False

            done, _ = await asyncio.wait(
                {speech_task, interrupt_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if interrupt_task in done and interrupt_task.result():
                self.controller.interrupt_speech()
                with suppress(Exception):
                    await speech_task
                print("\n⏭️ Speech interrupted. Listening again...")
                return True

            await speech_task
            return False
        finally:
            stop_event.set()
            if interrupt_task is not None:
                with suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(interrupt_task, timeout=0.3)

    async def check_ollama_model(self, session: aiohttp.ClientSession) -> bool:
        """Check if the requested model is available in Ollama."""
        try:
            print(f"🔍 Checking Ollama model '{self.model}'...")
            async with session.get(f"{self.ollama_url}/api/tags", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    # Check for exact match or match without tag (e.g. 'qwen2.5:0.5b' vs 'qwen2.5:0.5b-instruct')
                    # Ollama models usually have tags.
                    if self.model in models:
                        print(f"✅ Model '{self.model}' found.")
                        return True
                    # Check if 'latest' tag is implied
                    if f"{self.model}:latest" in models:
                        print(f"✅ Model '{self.model}:latest' found.")
                        return True
                        
                    print(f"⚠️ Model '{self.model}' not found in Ollama list.")
                    print(f"   Available models: {', '.join(models)}")
                    print("   Attempting to use it anyway (Ollama might pull it or error)...")
                    return False
        except Exception as e:
            print(f"⚠️ Could not check available models: {e}")
        return True  # Assume it might work

    async def _get_ollama_response_async(self, prompt: str, session: aiohttp.ClientSession) -> Optional[str]:
        """Get response from Ollama (streaming) using /api/chat with history."""
        try:
            if self.debug:
                print(f"\nDEBUG: Sending request to {self.ollama_url}/api/chat")
                print(f"DEBUG: Model: {self.model}")
                if self.history.enabled:
                    print(f"DEBUG: {self.history.get_summary()}")

            # Increase timeout significantly as loading a model can take time
            timeout_seconds = 300 
            
            # Build messages with history
            if self.history.enabled:
                # Get existing history (includes system prompt)
                messages = self.history.get_messages(include_system=True)
                # Add current user message
                messages.append({"role": "user", "content": prompt})
            else:
                # Original behavior without history
                messages = [
                    {"role": "system", "content": "You are a cute desktop robot assistant. Respond with enthusiasm and warmth."},
                    {"role": "user", "content": prompt}
                ]

            async with session.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model, 
                    "messages": messages,
                    "stream": True,
                    # Some thinking-capable models can emit only `message.thinking`.
                    # Ask for direct answer text in `message.content`.
                    "think": False,
                    "options": {"temperature": 0.8, "num_predict": 200}
                },
                timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"\n⚠️ Ollama error ({response.status}): {error_text}")
                    return None

                if self.debug:
                    print(f"DEBUG: Response received (Status {response.status}). Streaming content...")

                full_response = ""
                thinking_response = ""
                chunk_count = 0

                while True:
                    line = await response.content.readline()
                    if not line:
                        break
                    if line:
                        try:
                            decoded = line.decode('utf-8')
                            chunk = json.loads(decoded)
                            chunk_count += 1
                            
                            if self.debug and chunk_count <= 3:
                                print(f"DEBUG Chunk {chunk_count}: {decoded.strip()}")
                                
                            content = ""
                            # Handle /api/chat response format
                            if 'message' in chunk and 'content' in chunk['message']:
                                content = chunk['message']['content']
                                thinking_response += chunk['message'].get('thinking', '')
                            # Fallback for /api/generate format (just in case)
                            elif 'response' in chunk:
                                content = chunk['response']
                                
                            if content:
                                print(content, end="", flush=True)
                                full_response += content
                                
                            if chunk.get('done'):
                                if self.debug:
                                    print(f"\nDEBUG: Generation complete. Total stats: {chunk.get('total_duration', 0)/1e9:.2f}s")
                                
                        except Exception as e:
                            if self.debug:
                                print(f"\nDEBUG: JSON parse error: {e}")
                            continue
                
                if not full_response and thinking_response:
                    # Fallback for servers/models that still stream into `thinking`.
                    print(thinking_response, end="", flush=True)
                    full_response = thinking_response
                    if self.debug:
                        print("\nDEBUG: Used thinking stream as fallback response.")

                print()
                if not full_response and self.debug:
                    print("DEBUG: Warning - Empty response received from Ollama")
                    
                return full_response
                
        except asyncio.TimeoutError:
            print(f"\n⚠️ Ollama request timed out after {timeout_seconds}s")
            return None
        except Exception as e:
            print(f"\n⚠️ Ollama async error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

    async def _show_thinking_animation(self, reachy: ReachyMini, duration: float = 5.0):
        """Show robot 'thinking' animation."""
        import math
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                angle = math.sin((time.time() - start_time) * 3) * 0.1
                pose = create_head_pose(roll=angle)
                reachy.goto_target(head=pose, duration=0.3)
                await asyncio.sleep(0.1)

                if hasattr(reachy, 'l_antenna') and hasattr(reachy, 'r_antenna'):
                    reachy.l_antenna.goto_position(angle * 0.5, duration=0.2)
                    reachy.r_antenna.goto_position(-angle * 0.5, duration=0.2)

                await asyncio.sleep(0.2)
        finally:
            reachy.goto_target(head=create_head_pose(), duration=0.5)

    async def start_chat_async(self):
        print("="*60)
        print("🤖 Reachy Mini Chat v9 with Piper-TTS")
        print("="*60)
        print(f"Ollama Model: {self.model}")
        print(f"Ollama URL: {self.ollama_url}")
        print(f"Piper Model: {self.piper_model}")
        # Step 2: Show history status
        if self.history.enabled:
            print(f"💬 Conversation history: {self.history.max_rounds} rounds (type 'clear' to reset)")
        else:
            print("💬 Conversation history: disabled")
        print("💡 Need more voices? Download .onnx models from:")
        print("   https://github.com/rhasspy/piper/releases/tag/v0.0.2")

        try:
            with ReachyMini(media_backend="no_media") as reachy:
                print("✅ Connected to Reachy Mini")
                
                # Initialize controller with Piper
                self.controller = EmotionControllerV71(
                    reachy, 
                    self.piper_model, 
                    self.piper_config, 
                    self.speaker_id, 
                    self.debug,
                    gentle_mode=self.gentle
                )
                
                reachy.goto_target(head=create_head_pose(), duration=1.0)
                await asyncio.sleep(1.0)

                if self.use_asr:
                    if FasterWhisperASREngine is None:
                        print("❌ ASR requested but FasterWhisperASREngine not available.")
                        return

                    print(f"Initializing ASR engine ({self.asr_model}, VAD: {self.vad_silence}s silence)... (this may take a few seconds)")
                    try:
                        # Run ASR initialization in thread to not block event loop
                        self.asr_engine = await asyncio.to_thread(
                            FasterWhisperASREngine,
                            model_name=self.asr_model,
                            device='cpu'
                        )
                    except Exception as e:
                        print(f"❌ Failed to initialize ASR engine: {e}")
                        return

                    print("\n🎤 VAD ASR + Async mode: Ctrl-D skips speech, Ctrl-C exits")
                    
                    async with aiohttp.ClientSession() as session:
                        # Check model once
                        await self.check_ollama_model(session)
                        
                        while True:
                            try:
                                print("\n🎙️ Speak now... (Ctrl+C to exit)")
                                
                                # Step 4 A: Timing - ASR
                                asr_start = time.time()
                                
                                if self.use_vad:
                                    # VAD-based recording - stops on silence
                                    transcription = await asyncio.to_thread(
                                        self.asr_engine.transcribe_from_mic_vad,
                                        max_duration=4.0,
                                        silence_threshold=self.vad_silence,
                                        aggressiveness=self.vad_aggressive,
                                        trailing_buffer_ms=300
                                    )
                                else:
                                    # Fixed-duration recording - always records 4s
                                    transcription = await asyncio.to_thread(
                                        self.asr_engine.transcribe_from_mic,
                                        duration=4.0
                                    )
                                
                                asr_time = time.time() - asr_start
                                
                                if not transcription:
                                    print("⚠️ No speech detected, try again")
                                    continue

                                # Step 2: Add to history
                                self.history.add_user_message(transcription)
                                
                                print(f"📝 You: {transcription}")
                                if self.history.enabled:
                                    print(f"  {self.history.get_summary()}")
                                print("\n🤖 Reachy Mini: ", end="", flush=True)
                                
                                # Step 4 A: Timing - LLM
                                llm_start = time.time()
                                
                                thinking_task = asyncio.create_task(self._show_thinking_animation(reachy, 10.0))
                                llm_task = asyncio.create_task(self._get_ollama_response_async(transcription, session))
                                
                                response = await llm_task
                                
                                llm_time = time.time() - llm_start
                                
                                thinking_task.cancel()
                                with suppress(asyncio.CancelledError):
                                    await thinking_task
                                
                                if response and self.controller:
                                    # Step 2: Add assistant response to history
                                    self.history.add_assistant_message(response)
                                    
                                    # Step 4 A: Timing - TTS/Animation
                                    tts_start = time.time()
                                    
                                    emotion, intensity, emotion_level = self.controller.analyze_emotion(response)
                                    interrupted = await self._run_speech_with_interrupt(
                                        response, emotion, intensity, emotion_level
                                    )
                                    if interrupted:
                                        continue
                                      
                                    tts_time = time.time() - tts_start
                                    total_time = asr_time + llm_time + tts_time
                                    
                                    # Step 4 A: Display timing
                                    if self.debug:
                                        print(f"\n  ⏱️  [Timing] ASR: {asr_time:.2f}s, LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s, Total: {total_time:.2f}s")

                            except KeyboardInterrupt:
                                if self.controller:
                                    self.controller.interrupt_speech()
                                print("\n\n👋 Goodbye!")
                                return
                            except Exception as e:
                                print(f"⚠️ Error: {e}")
                                await asyncio.sleep(1.0)

                else:
                    print("\n💬 Start chatting (type 'quit' or Ctrl+C to exit)")
                    async with aiohttp.ClientSession() as session:
                        # Check model once
                        await self.check_ollama_model(session)
                        
                        while True:
                            try:
                                user_input = input("\n🧑 You: ").strip()
                                if user_input.lower() in ['quit', 'exit', 'q']:
                                    break
                                if user_input.lower() == 'clear':
                                    self.history.clear()
                                    continue
                                if not user_input:
                                    continue

                                # Step 2: Add to history
                                self.history.add_user_message(user_input)
                                
                                print("\n🤖 Reachy Mini: ", end="", flush=True)
                                
                                # Step 4 A: Timing - LLM
                                llm_start = time.time()
                                
                                thinking_task = asyncio.create_task(self._show_thinking_animation(reachy, 10.0))
                                llm_task = asyncio.create_task(self._get_ollama_response_async(user_input, session))
                                
                                response = await llm_task
                                
                                llm_time = time.time() - llm_start
                                
                                thinking_task.cancel()
                                with suppress(asyncio.CancelledError):
                                    await thinking_task
                                
                                if response and self.controller:
                                    # Step 2: Add assistant response to history
                                    self.history.add_assistant_message(response)
                                    
                                    # Step 4 A: Timing - TTS/Animation
                                    tts_start = time.time()
                                    
                                    emotion, intensity, emotion_level = self.controller.analyze_emotion(response)
                                    await self.controller.speak_with_expression_parallel(
                                        response, emotion, intensity, emotion_level
                                    )
                                     
                                    tts_time = time.time() - tts_start
                                    total_time = llm_time + tts_time
                                    
                                    # Step 4 A: Display timing
                                    if self.debug:
                                        print(f"\n  ⏱️  [Timing] LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s, Total: {total_time:.2f}s")

                            except KeyboardInterrupt:
                                print("\n\n👋 Goodbye!")
                                return
                            except Exception as e:
                                print(f"\n⚠️ Error: {e}")

        except Exception as e:
            print(f"\n❌ Cannot connect to Reachy Mini: {e}")
            self._tts_only_mode()

    def start_chat(self):
        asyncio.run(self.start_chat_async())

    def _tts_only_mode(self):
        print("\n📻 Running in TTS-only mode (no robot)")
        print("💡 Need more voices? Download .onnx models from:")
        print("   https://github.com/rhasspy/piper/releases/tag/v0.0.2")
        tts = PiperTTSEngine(self.piper_model, self.piper_config, self.speaker_id, self.debug)
        tts.speak_with_emotion("Hello! Piper TTS is working.", "neutral")


def main():
    parser = argparse.ArgumentParser(description="Reachy Mini Chat v9 with Piper-TTS and History")
    parser.add_argument('--chat', action='store_true', help='Start interactive chat')
    parser.add_argument('--asr', action='store_true', help='Use microphone ASR input')
    parser.add_argument('--model', default='qwen3:0.6b', help='Ollama model name (e.g., qwen2.5:0.5b)')
    parser.add_argument('--url', default='http://localhost:11434', help='Ollama URL')
    parser.add_argument('--piper-model', default='en_US-libritts_r-medium.onnx', help='Path to Piper .onnx model')
    parser.add_argument('--piper-config', default=None, help='Path to Piper .json config')
    parser.add_argument('--speaker', type=int, default=0, help='Speaker ID for multi-speaker models')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--gentle', action='store_true', help='Enable gentle_mode for subtle emotions')
    # Step 2: History options
    parser.add_argument('--history-size', type=int, default=5, help='Conversation history size (default: 5)')
    parser.add_argument('--no-history', action='store_true', help='Disable conversation history')
    # Step 4 B: ASR model selection
    parser.add_argument('--asr-model', default='small', choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='ASR model size: tiny=fastest, base=balanced, small=default, medium/large=slow but accurate')
    # VAD optimization: dynamic silence detection
    parser.add_argument('--vad-silence', type=float, default=0.8,
                        help='VAD silence threshold in seconds (default: 0.8). Increase if speech is cut off')
    parser.add_argument('--vad-aggressive', type=int, default=1, choices=[0, 1, 2, 3],
                        help='VAD aggressiveness: 0=least aggressive (more false positives), 1=gentle(recommended), 2=strict, 3=most aggressive (may cut speech)')
    parser.add_argument('--no-vad', action='store_true',
                        help='Disable VAD - use fixed 4s recording instead')

    args = parser.parse_args()
    
    # Needs aiohttp
    try:
        import aiohttp
    except ImportError:
        print("❌ aiohttp not found. Please install: pip install aiohttp")
        return

    app = ChatAppWithPiper(
        model=args.model, 
        ollama_url=args.url, 
        piper_model=args.piper_model,
        piper_config=args.piper_config,
        speaker_id=args.speaker,
        debug=args.debug, 
        use_asr=args.asr,
        gentle=args.gentle,
        history_size=args.history_size,
        enable_history=not args.no_history,
        asr_model=args.asr_model,
        vad_silence=args.vad_silence,
        vad_aggressive=args.vad_aggressive,
        use_vad=not args.no_vad
    )

    app.start_chat()


if __name__ == '__main__':
    main()
