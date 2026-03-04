#!/usr/bin/env python3
"""emo_v8_openclaw.py - Reachy Mini Chat with OpenClaw

基于 emo_v8.py，将 LLM 从 Ollama 替换为 OpenClaw。

Usage:
  python emo_v8_openclaw.py --token TOKEN --asr
  python emo_v8_openclaw.py --token TOKEN
"""

import time
import json
import asyncio
import aiohttp
import argparse
from typing import Optional

from emo_v6 import EmotionControllerV6, EdgeTTSEngine
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# Optional faster-whisper ASR engine
try:
    from utils.asr import FasterWhisperASREngine
except Exception:
    try:
        from .utils.asr import FasterWhisperASREngine
    except Exception:
        FasterWhisperASREngine = None


class ChatAppWithOpenClaw:
    def __init__(
        self,
        model: str = "bailian/qwen3.5-plus",
        openclaw_url: str = "http://127.0.0.1:18789",
        openclaw_token: str = None,
        debug: bool = False,
        use_asr: bool = False
    ):
        self.model = model
        self.openclaw_url = openclaw_url
        self.openclaw_token = openclaw_token
        self.debug = debug
        self.use_asr = use_asr
        self.controller: Optional[EmotionControllerV6] = None
        self.asr_engine = None
        
        # Validate that a token is provided
        if self.openclaw_token is None:
            raise ValueError("OpenClaw token required. Use --token argument.")

    async def _get_openclaw_response_async(
        self,
        prompt: str,
        session: aiohttp.ClientSession
    ) -> Optional[str]:
        """Get response from OpenClaw API."""
        try:
            # Use OpenAI-compatible API
            url = f"{self.openclaw_url}/v1/chat/completions"
            # Headers for OpenClaw API request
            payload = {
                "model": "openclaw",
                "messages": [
                    {"role": "system", "content": "You are a cute desktop "
                     "robot assistant. Respond with enthusiasm and warmth. "
                     "Keep responses short and friendly."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.8,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {self.openclaw_token}",
                "Content-Type": "application/json",
                "x-openclaw-agent-id": "main"
            }
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"\n⚠️ OpenClaw error ({response.status}): "
                          f"{error_text}")
                    return None
                
                data = await response.json()
                content = data.get("choices", [{}])[0].get(
                    "message", {}
                ).get("content", "")
                print(content, end="", flush=True)
                return content
                
        except asyncio.TimeoutError:
            print("\n⚠️ OpenClaw timeout")
            return None
        except Exception as e:
            print(f"\n⚠️ OpenClaw error: {e}")
            return None

    async def _get_openclaw_response_streaming_async(
        self,
        prompt: str,
        session: aiohttp.ClientSession
    ) -> Optional[str]:
        """Get streaming response from OpenClaw API."""
        try:
            url = f"{self.openclaw_url}/v1/chat/completions"
            # Headers for OpenClaw API request
            payload = {
                "model": "openclaw",
                "messages": [
                    {"role": "system", "content": "You are a cute desktop "
                     "robot assistant. Respond with enthusiasm and warmth. "
                     "Keep responses short and friendly."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.8,
                "stream": True
            }
            
            full_response = ""
            headers = {
                "Authorization": f"Bearer {self.openclaw_token}",
                "Content-Type": "application/json",
                "x-openclaw-agent-id": "main"
            }
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"\n⚠️ OpenClaw error ({response.status}): "
                          f"{error_text}")
                    return None
                
                async for line in response.content:
                    if line:
                        try:
                            text = line.decode('utf-8')
                            if text.startswith('data: '):
                                data = json.loads(text[6:])
                                if data.get("choices"):
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        print(content, end="", flush=True)
                                        full_response += content
                        except Exception:
                            continue
                
                print()
                return full_response
                
        except asyncio.TimeoutError:
            print("\n⚠️ OpenClaw timeout")
            return None
        except Exception as e:
            print(f"\n⚠️ OpenClaw error: {e}")
            return None

    async def _show_thinking_animation(
        self,
        reachy: ReachyMini,
        duration: float = 5.0
    ):
        """Show robot 'thinking' animation during LLM processing."""
        import math
        start_time = time.time()
        while time.time() - start_time < duration:
            # Gentle head movements to show "thinking"
            angle = math.sin((time.time() - start_time) * 3) * 0.1
            pose = create_head_pose(roll=angle)
            reachy.goto_target(head=pose, duration=0.3)
            await asyncio.sleep(0.1)
            
            # Small antenna movements
            if hasattr(reachy, 'l_antenna') and hasattr(reachy, 'r_antenna'):
                reachy.l_antenna.goto_position(angle * 0.5, duration=0.2)
                reachy.r_antenna.goto_position(-angle * 0.5, duration=0.2)
            
            await asyncio.sleep(0.2)
            
        # Return to neutral position
        reachy.goto_target(head=create_head_pose(), duration=0.5)

    async def start_chat_async(self):
        """Async version of chat with non-blocking LLM requests."""
        print("="*60)
        print("🤖 Reachy Mini Chat v8 with OpenClaw")
        print("="*60)
        print(f"Model: {self.model}")
        print(f"OpenClaw: {self.openclaw_url}")

        try:
            with ReachyMini(media_backend="no_media") as reachy:
                print("✅ Connected to Reachy Mini")
                self.controller = EmotionControllerV6(reachy, debug=self.debug)
                reachy.goto_target(head=create_head_pose(), duration=1.0)
                await asyncio.sleep(1.0)

                if self.use_asr:
                    if FasterWhisperASREngine is None:
                        print("❌ ASR not available. Install faster-whisper.")
                        return

                    print("Initializing ASR engine...")
                    try:
                        self.asr_engine = FasterWhisperASREngine(
                            model_name='small',
                            device='cpu'
                        )
                    except Exception as e:
                        print(f"❌ Failed to initialize ASR engine: {e}")
                        return

                    print("\n🎤 VAD ASR + Async mode: press Ctrl-C to stop")
                    print("🤔 Robot will show 'thinking' during LLM processing")
                    print("⏱️ Latency measurements enabled")
                    
                    async with aiohttp.ClientSession() as session:
                        while True:
                            try:
                                print("\n🎙️ Speak now...")
                                start_time = time.time()
                                transcription = self.asr_engine.transcribe_from_mic_vad(4.0, 1.5)
                                asr_latency = time.time() - start_time
                                
                                if not transcription:
                                    print("⚠️ No speech detected, try again")
                                    continue

                                print(f"\n⏱️ ASR latency: {asr_latency:.2f}s")
                                print(f"📝 You (transcribed): {transcription}")
                                
                                print("\n🤖 Reachy Mini: ", end="", flush=True)
                                
                                # Start thinking animation and LLM request
                                llm_start = time.time()
                                t = transcription
                                s = session
                                thinking_task = asyncio.create_task(
                                    self._show_thinking_animation(
                                        reachy,
                                        duration=10.0
                                    )
                                )
                                llm_task = asyncio.create_task(
                                    self._get_openclaw_response_streaming_async(t, s)
                                )
                                
                                # Wait for LLM response
                                response = await llm_task
                                llm_latency = time.time() - llm_start
                                
                                # Cancel thinking animation
                                thinking_task.cancel()
                                print(f"\n⏱️ LLM latency: {llm_latency:.2f}s")
                                
                                if response and self.controller:
                                    resp = response
                                    emotion, intensity, emotion_level = (
                                        self.controller.analyze_emotion(resp)
                                    )
                                    if self.debug:
                                        print(f"\n🎭 Emotion: {emotion}")
                                    # Use PARALLEL TTS
                                    controller = self.controller
                                    speak_method = controller.speak_with_expression_parallel
                                    await speak_method(response, emotion, intensity, emotion_level)

                            except KeyboardInterrupt:
                                print("\n👋 Exiting ASR chat")
                                break
                            except asyncio.CancelledError:
                                print("\n⚠️ Task cancelled")
                                break
                            except Exception as e:
                                print(f"⚠️ Error during ASR chat loop: {e}")
                                await asyncio.sleep(1.0)

                else:
                    print("\n💬 Start chatting (type 'quit' to exit)")
                    async with aiohttp.ClientSession() as session:
                        while True:
                            try:
                                user_input = input("\n🧑 You: ").strip()
                                if user_input.lower() in ['quit', 'exit', 'q']:
                                    break
                                if not user_input:
                                    continue

                                print("\n🤖 Reachy Mini: ", end="", flush=True)
                                
                                # Start thinking animation and LLM request
                                llm_start = time.time()
                                thinking_task = asyncio.create_task(
                                    self._show_thinking_animation(
                                        reachy,
                                        duration=10.0
                                    )
                                )
                                llm_task = asyncio.create_task(
                                    self._get_openclaw_response_streaming_async(user_input, session)
                                )
                                
                                response = await llm_task
                                llm_latency = time.time() - llm_start
                                
                                thinking_task.cancel()
                                print(f"\n⏱️ LLM latency: {llm_latency:.2f}s")
                                
                                if response and self.controller:
                                    resp = response
                                    emotion, intensity, emotion_level = (
                                        self.controller.analyze_emotion(resp)
                                    )
                                    if self.debug:
                                        print(f"\n🎭 Emotion: {emotion}")
                                    # Use PARALLEL TTS for text mode too
                                    controller = self.controller
                                    await controller.speak_with_expression_parallel(response, emotion, intensity, emotion_level)

                            except KeyboardInterrupt:
                                print("\n\n👋 Interrupted")
                                break
                            except Exception as e:
                                print(f"\n⚠️ Error: {e}")

        except Exception as e:
            print(f"\n❌ Cannot connect to Reachy Mini: {e}")
            print("Falling back to TTS-only mode")
            self._tts_only_mode()

    def start_chat(self):
        """Synchronous wrapper for backward compatibility."""
        asyncio.run(self.start_chat_async())

    def _tts_only_mode(self):
        print("\n📻 Running in TTS-only mode (no robot)")
        try:
            with ReachyMini(media_backend="no_media") as reachy:
                controller = EmotionControllerV6(reachy, debug=self.debug)

                test_texts = [
                    ("你好！我是Reachy Mini！", "neutral"),
                    ("我今天很开心！", "positive"),
                    ("我有点难过...", "negative"),
                    ("这是什么？", "question"),
                ]

                for text, emotion in test_texts:
                    print(f"\nTesting: '{text}'")
                    controller.tts_engine.speak_with_emotion(text, emotion)
                    time.sleep(1.0)

        except Exception:
            print("\nTesting Edge-TTS standalone...")
            tts_engine = EdgeTTSEngine()
            tts_engine.speak_with_emotion(
                "Hello! This is Edge-TTS working.", "neutral"
            )


def main():
    parser = argparse.ArgumentParser(description="Reachy Mini Chat OpenClaw")
    parser.add_argument('--chat', action='store_true', help='Start chat')
    parser.add_argument('--asr', action='store_true', help='Use ASR input')
    parser.add_argument('--model', default='bailian/qwen3.5-plus', help='')
    parser.add_argument('--url', default='http://127.0.0.1:18789', help='URL')
    parser.add_argument('--token', default=None, help='Auth token (required)')
    parser.add_argument('--debug', action='store_true', help='Enable debug')

    args = parser.parse_args()
    if args.token is None:
        print("Error: OpenClaw token required. Use --token.")
        print("Example: python emo_v8_openclaw.py --token YOUR_TOKEN")
        return
    
    app = ChatAppWithOpenClaw(
        model=args.model,
        openclaw_url=args.url,
        openclaw_token=args.token,
        debug=args.debug,
        use_asr=args.asr
    )

    app.start_chat()


if __name__ == '__main__':
    main()
