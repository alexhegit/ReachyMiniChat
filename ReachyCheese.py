#!/usr/bin/env python3
"""ReachyCheese - offline voice-interactive photo app for Reachy Mini.

Flow:
1. Sleep: listen wake word ("Reachy")
2. Tracking: align largest face to center (head-first + body compensation)
3. Armed: wait capture phrase ("cheese", "take photo", "take picture")
4. Countdown: "Look at me. Hold still... Ready? One, two, three, cheese!"
5. Capture one photo and save to ~/Pictures/ReachyMiniPhoto/
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import queue
import re
import smtplib
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from enum import Enum
from mimetypes import guess_type
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose
except Exception:  # pragma: no cover - optional runtime dependency
    ReachyMini = None  # type: ignore[assignment]

    def create_head_pose(*args, **kwargs):  # type: ignore[no-redef]
        return None

from vision.face_tracker import FaceTracker
from utils.asr import FasterWhisperASREngine

try:
    from emo_v9 import PiperTTSEngine
except Exception:  # pragma: no cover - optional runtime dependency
    PiperTTSEngine = None  # type: ignore[assignment]


class RCState(str, Enum):
    SLEEP = "sleep"
    TRACKING = "tracking"
    ARMED = "armed"
    COUNTDOWN = "countdown"


@dataclass
class EmailConfig:
    recipient: str = ""
    sender: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_encryption: str = "start-tls"  # start-tls | tls | none
    subject: str = "Reachy camera photo"
    body: str = "Hi,\n\nAttached is the latest Reachy camera photo.\n\n- ReachyCheese"

    @property
    def enabled(self) -> bool:
        return bool(self.recipient.strip() and self.sender.strip() and self.smtp_password.strip())


class PhotoEmailSender:
    def __init__(self, cfg: EmailConfig):
        self.cfg = cfg

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.sender.strip() and self.cfg.smtp_password.strip())

    def send_photo(self, photo_path: Path, recipient_override: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        recipient = (recipient_override or self.cfg.recipient).strip()
        if not self.enabled:
            return False, "email sender is not enabled"
        if not recipient:
            return False, "recipient is empty"
        if not photo_path.exists():
            return False, f"photo not found: {photo_path}"

        msg = EmailMessage()
        msg["From"] = self.cfg.sender
        msg["To"] = recipient
        msg["Subject"] = self.cfg.subject
        msg.set_content(self.cfg.body)

        ctype, encoding = guess_type(str(photo_path))
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)

        with open(photo_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype=maintype,
                subtype=subtype,
                filename=photo_path.name,
            )

        username = self.cfg.smtp_username.strip() or self.cfg.sender.strip()
        encryption = (self.cfg.smtp_encryption or "start-tls").strip().lower()
        context = ssl.create_default_context()

        try:
            if encryption == "tls":
                with smtplib.SMTP_SSL(self.cfg.smtp_host, self.cfg.smtp_port, timeout=30, context=context) as server:
                    if self.cfg.smtp_password:
                        server.login(username, self.cfg.smtp_password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port, timeout=30) as server:
                    server.ehlo()
                    if encryption == "start-tls":
                        server.starttls(context=context)
                        server.ehlo()
                    if self.cfg.smtp_password:
                        server.login(username, self.cfg.smtp_password)
                    server.send_message(msg)
        except Exception as exc:
            return False, str(exc)

        return True, None


@dataclass
class RCConfig:
    preview_width: int = 640
    preview_height: int = 480
    preview_fps: float = 20.0
    save_dir: Path = Path.home() / "Pictures" / "ReachyMiniPhoto"
    wake_word: str = "reachy"
    command_timeout_s: float = 12.0
    asr_mode: str = "disable"  # disable | vad
    asr_model: str = "base"
    vad_silence: float = 0.7
    vad_aggressive: int = 1
    piper_model: str = "models/en-us-blizzard_lessac-medium.onnx"
    piper_config: Optional[str] = None
    speaker_id: int = 0
    camera_source: str = "reachy"
    camera_index: int = 0
    controller: str = "v2"
    debug: bool = False
    track_log_path: Optional[Path] = None
    email: EmailConfig = field(default_factory=EmailConfig)


@dataclass
class FaceTrackStatus:
    has_face: bool
    aligned: bool
    bbox: Optional[Tuple[int, int, int, int]]
    center: Optional[Tuple[int, int]]
    dx: float
    dy: float
    stable_frames: int


class RobotRuntime:
    def get_frame(self):
        raise NotImplementedError

    def look_at_image(self, x: int, y: int, duration: float = 0.2) -> None:
        raise NotImplementedError

    def goto_body_yaw(self, yaw: float, duration: float = 0.35) -> None:
        raise NotImplementedError

    def reset_head(self, duration: float = 0.2) -> None:
        raise NotImplementedError

    def set_automatic_body_yaw(self, enabled: bool) -> None:
        raise NotImplementedError

    def set_antennas(self, left: float, right: float, duration: float = 0.2) -> None:
        raise NotImplementedError

    def set_head_pitch(self, pitch: float, duration: float = 0.2) -> None:
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class ReachyRuntime(RobotRuntime):
    def __init__(self):
        self._ctx = None
        self._reachy = None

    def __enter__(self):
        if ReachyMini is None:
            raise RuntimeError("reachy_mini is not available. Use --camera-source webcam for local test.")
        self._ctx = ReachyMini(media_backend="default")
        self._reachy = self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc_val, exc_tb)
        return False

    def get_frame(self):
        if self._reachy and hasattr(self._reachy, "media") and self._reachy.media:
            return self._reachy.media.get_frame()
        return None

    def look_at_image(self, x: int, y: int, duration: float = 0.2) -> None:
        if self._reachy:
            self._reachy.look_at_image(x, y, duration=duration)

    def goto_body_yaw(self, yaw: float, duration: float = 0.35) -> None:
        if self._reachy:
            self._reachy.goto_target(body_yaw=yaw, duration=duration)

    def reset_head(self, duration: float = 0.2) -> None:
        if self._reachy:
            self._reachy.goto_target(head=create_head_pose(), duration=duration)

    def set_automatic_body_yaw(self, enabled: bool) -> None:
        if self._reachy:
            self._reachy.set_automatic_body_yaw(enabled)

    def set_antennas(self, left: float, right: float, duration: float = 0.2) -> None:
        if self._reachy:
            self._reachy.goto_target(antennas=[float(left), float(right)], duration=duration)

    def set_head_pitch(self, pitch: float, duration: float = 0.2) -> None:
        if self._reachy:
            self._reachy.goto_target(head=create_head_pose(pitch=float(pitch)), duration=duration)


class WebcamRuntime(RobotRuntime):
    def __init__(self, camera_index: int):
        self._camera_index = camera_index
        self._cap = None

    def __enter__(self):
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {self._camera_index}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap is not None:
            self._cap.release()
        return False

    def get_frame(self):
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def look_at_image(self, x: int, y: int, duration: float = 0.2) -> None:
        return

    def goto_body_yaw(self, yaw: float, duration: float = 0.35) -> None:
        return

    def reset_head(self, duration: float = 0.2) -> None:
        return

    def set_automatic_body_yaw(self, enabled: bool) -> None:
        return

    def set_antennas(self, left: float, right: float, duration: float = 0.2) -> None:
        return

    def set_head_pitch(self, pitch: float, duration: float = 0.2) -> None:
        return


class VoiceIO:
    def __init__(self, cfg: RCConfig):
        self.cfg = cfg
        self._speak_queue: "queue.Queue[str]" = queue.Queue()
        self._speak_running = True
        self._speak_thread = threading.Thread(target=self._speak_loop, daemon=True)
        self._tts = None
        if PiperTTSEngine is not None:
            self._tts = PiperTTSEngine(
                model_path=cfg.piper_model,
                config_path=cfg.piper_config,
                speaker_id=cfg.speaker_id,
                debug=cfg.debug,
            )
        self._asr = FasterWhisperASREngine(model_name=cfg.asr_model, device="cpu")
        self._speak_thread.start()

    def close(self) -> None:
        self._speak_running = False
        self._speak_queue.put("")
        self._speak_thread.join(timeout=1.5)

    def _speak_loop(self) -> None:
        while self._speak_running:
            text = self._speak_queue.get()
            if not text:
                continue
            try:
                if self._tts and getattr(self._tts, "voice", None):
                    self._tts.speak_with_emotion(text, "neutral")
                else:
                    print(f"🔊 {text}")
            except Exception as exc:
                print(f"⚠️ TTS error: {exc}")

    def speak_async(self, text: str) -> None:
        if text.strip():
            self._speak_queue.put(text.strip())

    def listen_once(self) -> str:
        text = self._asr.transcribe_from_mic_vad(
            max_duration=4.5,
            silence_threshold=self.cfg.vad_silence,
            aggressiveness=self.cfg.vad_aggressive,
            trailing_buffer_ms=400,
            show_volume=False,
        )
        heard = (text or "").strip().lower()
        if heard:
            return heard
        # Fallback when VAD cuts too early.
        fallback = self._asr.transcribe_from_mic(duration=2.5)
        return (fallback or "").strip().lower()


class FaceAlignerV2:
    def __init__(self):
        self._tracker = FaceTracker(
            smooth_factor=0.20,
            multi_face_strategy="largest",
            min_detection_confidence=0.70,
        )
        # Horizontal can be looser, but vertical should stay tighter to avoid
        # getting stuck in "head-up" poses without downward correction.
        self._deadzone_x = 32
        self._deadzone_y = 12
        self._lock_hold_x = 44
        self._lock_hold_y = 18
        self._release_x = 52
        self._release_y = 20
        self._stable_needed = 6
        self._stable_frames = 0
        self._ema_dx = 0.0
        self._ema_dy = 0.0
        self._ema_dy_rate = 0.0
        self._last_ema_dy = 0.0
        self._last_ema_t = 0.0
        self._alpha = 0.30
        self._last_track_at = 0.0
        self._big_error_since = 0.0
        self._body_yaw = 0.0
        self._max_body_yaw = 0.8
        self._locked = False
        self._last_cmd_center: Optional[Tuple[int, int]] = None
        self._last_cmd_sent_at = 0.0
        self._last_dy_sign = 0
        self._min_cmd_delta_px = 24
        # When vertical error persists at a large value, periodically resend
        # look_at commands even if target pixel barely changes. This avoids
        # getting stuck in a head-up/down posture due to command deduping.
        self._force_resend_dy = 48
        self._force_resend_interval_s = 0.72
        self._reacquire_x = 170
        self._reacquire_y = 130
        self._cmd_max_step_x = 95
        self._cmd_max_step_y = 100
        self._target_x_gain = 0.55
        # Vertical uses asymmetric gains/limits to avoid downward overshoot:
        # - dy > 0 (face below center): pitch down, but gentler
        # - dy < 0 (face above center): pitch up, can be stronger for recovery
        self._target_y_gain_down = 0.36
        self._target_y_gain_up = 0.72
        self._target_y_max_step_down = 36
        self._target_y_max_step_up = 92
        # Large vertical error recovery profile (|dy| very large):
        # use stronger authority to pull back from +300/-300 style offsets,
        # then fall back to the gentle profile near center to avoid overshoot.
        self._vertical_reacquire_dy = 120
        self._target_y_gain_down_far = 0.94
        self._target_y_gain_up_far = 0.90
        self._target_y_max_step_down_far = 136
        self._target_y_max_step_up_far = 130
        # Ultra-large error profile ("turbo"), used only for very large vertical
        # offsets so recovery starts decisively before switching back to far/near.
        self._vertical_turbo_dy = 240
        self._target_y_gain_down_turbo = 1.10
        self._target_y_gain_up_turbo = 1.00
        self._target_y_max_step_down_turbo = 170
        self._target_y_max_step_up_turbo = 150
        # Vertical command sign is hardware/SDK dependent; on the current robot
        # setup, positive dy (face below center) should increase target y.
        self._target_y_sign = 1.0
        # New Y-axis direct pitch servo (decoupled from look_at_image Y channel).
        self._use_direct_pitch_y = True
        self._pitch_y_sign = 1.0
        self._head_pitch_cmd = 0.0
        self._head_pitch_min = -0.85
        self._head_pitch_max = 0.85
        self._pitch_kp_near = 0.0026
        self._pitch_kp_far = 0.0044
        self._pitch_ki = 0.00012
        self._pitch_kd = 0.00035
        self._pitch_i = 0.0
        self._pitch_i_clamp = 0.22
        self._pitch_max_delta = 0.09
        self._pitch_turbo_dy = 210
        self._pitch_turbo_scale = 1.3
        self._pitch_recenter_dy = 100
        self._last_pitch_cmd_at = 0.0
        self._settle_until = 0.0
        self._body_cooldown_until = 0.0
        # Two-stage follow policy with hysteresis:
        # 1) large horizontal error -> prioritize body recenter
        # 2) once below exit threshold -> head fine alignment
        self._body_recenter_threshold_x = 220
        self._body_recenter_exit_x = 150
        self._body_recenter_active = False

    def reset(self) -> None:
        self._stable_frames = 0
        self._ema_dx = 0.0
        self._ema_dy = 0.0
        self._ema_dy_rate = 0.0
        self._last_ema_dy = 0.0
        self._last_ema_t = 0.0
        self._last_track_at = 0.0
        self._big_error_since = 0.0
        self._body_yaw = 0.0
        self._locked = False
        self._last_cmd_center = None
        self._last_cmd_sent_at = 0.0
        self._last_dy_sign = 0
        self._pitch_i = 0.0
        self._head_pitch_cmd = 0.0
        self._last_pitch_cmd_at = 0.0
        self._settle_until = 0.0
        self._body_cooldown_until = 0.0
        self._body_recenter_active = False

    def update(self, runtime: RobotRuntime, frame, soft: bool = False) -> FaceTrackStatus:
        bbox = self._tracker.detect(frame)
        if bbox is None:
            self._stable_frames = 0
            self._locked = False
            return FaceTrackStatus(False, False, None, None, 0.0, 0.0, 0)

        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]
        # More permissive threshold so desktop-distance faces are still tracked.
        # Keep area gating only for alignment/capture stability, not for drawing/tracking feedback.
        min_face_area = int(frame_w * frame_h * 0.003)
        face_too_small = (w * h) < min_face_area

        cx, cy = x + (w // 2), y + (h // 2)
        dx = float(cx - frame_w // 2)
        dy = float(cy - frame_h // 2)
        now = time.time()

        self._ema_dx = self._alpha * dx + (1 - self._alpha) * self._ema_dx
        self._ema_dy = self._alpha * dy + (1 - self._alpha) * self._ema_dy
        if self._last_ema_t > 0.0:
            dt = max(1e-3, now - self._last_ema_t)
            raw_rate = (self._ema_dy - self._last_ema_dy) / dt
            # Smooth dy rate to avoid jittery derivative actions.
            self._ema_dy_rate = 0.35 * raw_rate + 0.65 * self._ema_dy_rate
        self._last_ema_dy = self._ema_dy
        self._last_ema_t = now
        curr_dy_sign = 1 if self._ema_dy > 0 else (-1 if self._ema_dy < 0 else 0)

        aligned_now = (not face_too_small) and abs(self._ema_dx) <= self._deadzone_x and abs(self._ema_dy) <= self._deadzone_y
        self._stable_frames = self._stable_frames + 1 if aligned_now else 0
        aligned = self._stable_frames >= self._stable_needed

        if aligned:
            self._locked = True
        elif self._locked:
            still_hold = abs(self._ema_dx) <= self._lock_hold_x and abs(self._ema_dy) <= self._lock_hold_y
            if not still_hold:
                self._locked = False

        in_reacquire = abs(self._ema_dx) > self._reacquire_x or abs(self._ema_dy) > self._reacquire_y
        move_interval = 0.28 if soft else (0.22 if in_reacquire else 0.16)
        self._diag_move_interval_s = move_interval
        need_move = abs(self._ema_dx) > self._release_x or abs(self._ema_dy) > self._release_y

        # Stage selection by horizontal error magnitude with hysteresis.
        # Stage 1 (body recenter): enter when |dx| > enter_threshold.
        # Stage 2 (head align): exit body stage when |dx| <= exit_threshold.
        abs_dx = abs(self._ema_dx)
        if self._body_recenter_active:
            if abs_dx <= self._body_recenter_exit_x:
                self._body_recenter_active = False
        else:
            if abs_dx > self._body_recenter_threshold_x:
                self._body_recenter_active = True

        large_horizontal_error = self._body_recenter_active
        self._diag_settle_remaining_s = max(0.0, self._settle_until - now)
        # During settle window after body motion, still allow emergency vertical
        # correction when |dy| is very large.
        if now < self._settle_until and abs(self._ema_dy) < self._vertical_reacquire_dy:
            return FaceTrackStatus(
                has_face=True,
                aligned=aligned,
                bbox=(x, y, w, h),
                center=(cx, cy),
                dx=self._ema_dx,
                dy=self._ema_dy,
                stable_frames=self._stable_frames,
            )

        if now - self._last_track_at >= move_interval:
            self._last_track_at = now

            did_body_move = False
            if need_move and not self._locked and large_horizontal_error and now >= self._body_cooldown_until:
                if self._big_error_since == 0.0:
                    self._big_error_since = now
                elif now - self._big_error_since > (0.45 if in_reacquire else 0.55):
                    # body_yaw sign should follow camera pixel convention from look_at_image:
                    # when face is on the right (dx > 0), head yaw goes negative to track it,
                    # so body compensation must also move in the negative direction.
                    step = -0.07 if self._ema_dx > 0 else 0.07
                    self._body_yaw = max(-self._max_body_yaw, min(self._max_body_yaw, self._body_yaw + step))
                    try:
                        runtime.goto_body_yaw(self._body_yaw, duration=0.45)
                    except Exception:
                        pass
                    self._big_error_since = 0.0
                    self._body_cooldown_until = now + 0.95
                    self._settle_until = now + 0.18
                    self._last_cmd_center = None
                    did_body_move = True
            else:
                self._big_error_since = 0.0

            # If no body move is issued in this tick, clear settle so head Y correction
            # is not unnecessarily blocked when dy remains large.
            if not did_body_move:
                self._settle_until = 0.0

            # Head alignment policy:
            # - Normal case: full X/Y head alignment when not in body-recenter stage.
            # - During body-recenter stage: still allow Y-only head correction so large
            #   vertical errors (e.g. dy +300) can be corrected instead of waiting for
            #   horizontal recenter to finish.
            vertical_priority_in_body_stage = large_horizontal_error and abs(self._ema_dy) > (self._release_y + 28)
            allow_head_cmd = (not large_horizontal_error) or vertical_priority_in_body_stage
            if need_move and not self._locked and not did_body_move and allow_head_cmd:
                if large_horizontal_error:
                    target_x = frame_w // 2
                else:
                    target_x = frame_w // 2 + int(np.clip(self._ema_dx * self._target_x_gain, -self._cmd_max_step_x, self._cmd_max_step_x))

                if self._use_direct_pitch_y:
                    abs_dy = abs(self._ema_dy)
                    if abs_dy >= self._pitch_turbo_dy:
                        kp_scale = self._pitch_turbo_scale
                    elif abs_dy >= self._pitch_recenter_dy:
                        kp_scale = 1.25
                    else:
                        kp_scale = 1.0

                    kp_base = self._pitch_kp_far if abs_dy >= self._pitch_recenter_dy else self._pitch_kp_near
                    kp = kp_base * kp_scale
                    kd = self._pitch_kd * kp_scale
                    err = self._pitch_y_sign * self._ema_dy
                    self._pitch_i += err * max(1e-3, move_interval)
                    self._pitch_i = float(np.clip(self._pitch_i, -self._pitch_i_clamp, self._pitch_i_clamp))
                    pitch_delta = kp * err + self._pitch_ki * self._pitch_i + kd * self._pitch_y_sign * self._ema_dy_rate
                    pitch_delta = float(np.clip(pitch_delta, -self._pitch_max_delta, self._pitch_max_delta))
                    self._head_pitch_cmd = float(np.clip(self._head_pitch_cmd + pitch_delta, self._head_pitch_min, self._head_pitch_max))

                    should_send = True
                    if self._last_cmd_center:
                        dcmd_x = abs(target_x - self._last_cmd_center[0])
                        x_thresh = self._min_cmd_delta_px if not large_horizontal_error else (self._min_cmd_delta_px * 2)
                        if dcmd_x < x_thresh and abs_dy < self._force_resend_dy:
                            should_send = False
                            if self._last_dy_sign != 0 and curr_dy_sign != 0 and curr_dy_sign != self._last_dy_sign:
                                should_send = True
                    if (now - self._last_pitch_cmd_at) >= self._force_resend_interval_s and abs_dy >= self._force_resend_dy:
                        should_send = True

                    if should_send:
                        try:
                            runtime.look_at_image(target_x, frame_h // 2, duration=0.34 if (soft or in_reacquire) else 0.24)
                            runtime.set_head_pitch(self._head_pitch_cmd, duration=0.22 if (soft or in_reacquire) else 0.18)
                            self._last_cmd_center = (target_x, frame_h // 2)
                            self._last_cmd_sent_at = now
                            self._last_pitch_cmd_at = now
                            self._last_dy_sign = curr_dy_sign
                            self._diag_last_cmd_age_s = 0.0
                        except Exception:
                            pass
                else:
                    abs_dy = abs(self._ema_dy)
                    if self._ema_dy >= 0:
                        # Face below center -> pitch down.
                        if abs_dy >= self._vertical_turbo_dy:
                            # Ultra-large recovery: fastest pull-down for dy~300+
                            y_gain = self._target_y_gain_down_turbo
                            y_step_max = self._target_y_max_step_down_turbo
                        elif abs_dy >= self._vertical_reacquire_dy:
                            # Far-from-center recovery.
                            y_gain = self._target_y_gain_down_far
                            y_step_max = self._target_y_max_step_down_far
                        else:
                            # Near-center precision: conservative to reduce overshoot.
                            y_gain = self._target_y_gain_down
                            y_step_max = self._target_y_max_step_down
                    else:
                        # Face above center -> pitch up.
                        if abs_dy >= self._vertical_turbo_dy:
                            y_gain = self._target_y_gain_up_turbo
                            y_step_max = self._target_y_max_step_up_turbo
                        elif abs_dy >= self._vertical_reacquire_dy:
                            y_gain = self._target_y_gain_up_far
                            y_step_max = self._target_y_max_step_up_far
                        else:
                            y_gain = self._target_y_gain_up
                            y_step_max = self._target_y_max_step_up
                    target_y = frame_h // 2 + int(np.clip(self._target_y_sign * self._ema_dy * y_gain, -y_step_max, y_step_max))

                    # Motion-lag compensation (predictive brake): if current dy is large
                    # but changing rapidly toward center, reduce command amplitude to avoid
                    # overshoot caused by perception-action delay.
                    dy_rate = self._ema_dy_rate
                    if curr_dy_sign != 0 and (dy_rate * curr_dy_sign) < -35.0:
                        # approaching center quickly -> clamp to gentler move
                        brake_scale = 0.55
                        target_y = frame_h // 2 + int((target_y - frame_h // 2) * brake_scale)
                    should_send = True
                    if self._last_cmd_center:
                        dcmd_x = abs(target_x - self._last_cmd_center[0])
                        dcmd_y = abs(target_y - self._last_cmd_center[1])
                        x_thresh = self._min_cmd_delta_px if not large_horizontal_error else (self._min_cmd_delta_px * 2)
                        y_thresh = self._min_cmd_delta_px
                        if dcmd_x < x_thresh and dcmd_y < y_thresh:
                            should_send = False
                            # If vertical correction direction just changed (e.g. overshoot
                            # from below-center to above-center), force one immediate send
                            # to brake the previous motion quickly.
                            if self._last_dy_sign != 0 and curr_dy_sign != 0 and curr_dy_sign != self._last_dy_sign:
                                should_send = True
                            # If vertical error remains significant, periodically re-send
                            # to prevent actuator stiction / posture lock on real hardware.
                            elif abs(self._ema_dy) >= self._force_resend_dy and (now - self._last_cmd_sent_at) >= self._force_resend_interval_s:
                                should_send = True
                    if should_send:
                        try:
                            runtime.look_at_image(target_x, target_y, duration=0.34 if (soft or in_reacquire) else 0.24)
                            self._last_cmd_center = (target_x, target_y)
                            self._last_cmd_sent_at = now
                            self._last_dy_sign = curr_dy_sign
                            self._diag_last_cmd_age_s = 0.0
                        except Exception:
                            pass

            if not need_move:
                self._last_cmd_center = None
                self._last_cmd_sent_at = 0.0
                self._last_dy_sign = 0
                self._pitch_i = 0.0
                self._diag_last_cmd_age_s = 0.0

        if self._last_cmd_sent_at > 0.0:
            self._diag_last_cmd_age_s = max(0.0, now - self._last_cmd_sent_at)
        else:
            self._diag_last_cmd_age_s = 0.0

        return FaceTrackStatus(
            has_face=True,
            aligned=aligned,
            bbox=(x, y, w, h),
            center=(cx, cy),
            dx=self._ema_dx,
            dy=self._ema_dy,
            stable_frames=self._stable_frames,
        )


class FaceAlignerV1(FaceAlignerV2):
    """Legacy compatibility alias for controller selection.

    Kept so users can force v1 behavior while v2 remains default.
    """


FaceAligner = FaceAlignerV2


class PreviewGUI:
    def __init__(self, cfg: RCConfig, event_queue: "queue.Queue[str]"):
        self.cfg = cfg
        self._events = event_queue
        self._ready = True
        self._window_name = "ReachyCheese"
        self._button_height = 44
        self._buttons = [
            ("Wake", "manual_wake"),
            ("Take Photo", "manual_capture"),
            ("Cancel", "manual_cancel"),
            ("Sleep", "manual_sleep"),
        ]

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, cfg.preview_width, cfg.preview_height + self._button_height + 60)
        cv2.setMouseCallback(self._window_name, self._on_mouse)

    @property
    def available(self) -> bool:
        return self._ready

    def is_running(self) -> bool:
        if not self._ready:
            return False
        try:
            visible = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE)
        except Exception:
            self._ready = False
            return False
        if visible < 1:
            self._ready = False
            return False
        return True

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            pass
        self._ready = False

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        top = self.cfg.preview_height + 10
        if y < top:
            return
        spacing = 8
        available_width = self.cfg.preview_width - spacing * (len(self._buttons) + 1)
        btn_w = available_width // len(self._buttons)
        for i, (_, action) in enumerate(self._buttons):
            bx = spacing + i * (btn_w + spacing)
            by = top
            if bx <= x <= bx + btn_w and by <= y <= by + self._button_height:
                self._events.put(action)
                return

    def draw(
        self,
        frame_bgr,
        state: RCState,
        hint: str,
        last_saved: str,
        status: Optional[FaceTrackStatus] = None,
        countdown_text: str = "",
        diag_line: str = "",
    ) -> None:
        src_h, src_w = frame_bgr.shape[:2]
        frame = cv2.resize(frame_bgr, (self.cfg.preview_width, self.cfg.preview_height))
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (0, 220, 220), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)

        if status and status.bbox:
            x, y, bw, bh = status.bbox
            sx = self.cfg.preview_width / float(src_w)
            sy = self.cfg.preview_height / float(src_h)
            px = int(x * sx)
            py = int(y * sy)
            pw = int(bw * sx)
            ph = int(bh * sy)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (50, 230, 50), 2)
            cv2.putText(
                frame,
                f"dx={status.dx:+.0f}, dy={status.dy:+.0f}",
                (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 60),
                2,
            )

        if countdown_text:
            cv2.putText(
                frame,
                countdown_text,
                (14, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (20, 20, 240),
                3,
            )

        panel_h = self._button_height + 60
        canvas = np.zeros((self.cfg.preview_height + panel_h, self.cfg.preview_width, 3), dtype=np.uint8)
        canvas[: self.cfg.preview_height, :, :] = frame

        cv2.putText(canvas, f"State: {state.value.upper()}", (10, self.cfg.preview_height + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2)
        cv2.putText(canvas, hint[:70], (10, self.cfg.preview_height + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        if status and status.has_face:
            face_line = f"Face stable={status.stable_frames}, aligned={status.aligned}"
        else:
            face_line = "Face: not detected"
        cv2.putText(canvas, face_line[:70], (320, self.cfg.preview_height + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 120), 1)
        if diag_line:
            cv2.putText(canvas, diag_line[:70], (320, self.cfg.preview_height + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 220, 255), 1)
        else:
            cv2.putText(canvas, f"Saved: {(last_saved or '--')[-60:]}", (320, self.cfg.preview_height + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 220, 160), 1)

        spacing = 8
        top = self.cfg.preview_height + 10
        available_width = self.cfg.preview_width - spacing * (len(self._buttons) + 1)
        btn_w = available_width // len(self._buttons)
        for i, (label, _) in enumerate(self._buttons):
            bx = spacing + i * (btn_w + spacing)
            by = top
            cv2.rectangle(canvas, (bx, by), (bx + btn_w, by + self._button_height), (70, 70, 70), -1)
            cv2.rectangle(canvas, (bx, by), (bx + btn_w, by + self._button_height), (130, 130, 130), 1)
            cv2.putText(canvas, label, (bx + 10, by + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 2)

        cv2.imshow(self._window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self._events.put("quit")


class ReachyCheeseApp:
    def __init__(self, cfg: RCConfig):
        self.cfg = cfg
        self.state = RCState.SLEEP
        self.voice = VoiceIO(cfg)
        self.aligner = FaceAligner() if cfg.controller == "v2" else FaceAlignerV1()
        self._event_queue: "queue.Queue[str]" = queue.Queue()
        self.gui = PreviewGUI(cfg, self._event_queue)
        self._asr_queue: "queue.Queue[str]" = queue.Queue()
        self._listener_running = False
        self._listener_thread: Optional[threading.Thread] = None
        self._last_frame = None
        self._hint = "Say 'Reachy' to wake"
        self._last_saved_path = ""
        self._diag_last_cmd_age_s = 0.0
        self._diag_move_interval_s = 0.0
        self._diag_settle_remaining_s = 0.0
        self._runtime: Optional[RobotRuntime] = None
        self._track_log_fp = None
        self._armed_since = 0.0
        self._email_sender = PhotoEmailSender(cfg.email)
        self._pending_email_recipient: Optional[str] = None

        self._countdown_started_at = 0.0
        self._countdown_index = 0
        self._countdown_lines = [
            (0.0, "Look at me. Hold still... Ready?"),
            (0.8, "One"),
            (1.6, "Two"),
            (2.4, "Three"),
            (3.2, "Cheese"),
        ]
        self._countdown_overlay = ""
        self._countdown_antenna_beats = {"one", "two", "three", "cheese"}
        self._countdown_antenna_phase = False
        self._countdown_started_by_manual = False
        self._countdown_allow_auto_cancel = False
        self._countdown_cancel_grace_s = 0.0
        self._countdown_outlier_streak = 0
        self._countdown_cancel_streak_need = 4

    def _open_track_log(self) -> None:
        if not self.cfg.track_log_path:
            return
        try:
            self.cfg.track_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._track_log_fp = open(self.cfg.track_log_path, "a", encoding="utf-8")
            print(f"📝 Tracking log: {self.cfg.track_log_path}")
        except Exception as exc:
            self._track_log_fp = None
            print(f"⚠️ Failed to open track log {self.cfg.track_log_path}: {exc}")

    def _close_track_log(self) -> None:
        if self._track_log_fp is not None:
            try:
                self._track_log_fp.close()
            except Exception:
                pass
            self._track_log_fp = None

    def _append_track_log(self, status: Optional[FaceTrackStatus]) -> None:
        if self._track_log_fp is None:
            return
        row = {
            "t": time.time(),
            "state": str(self.state.value),
            "has_face": bool(status.has_face) if status else False,
            "aligned": bool(status.aligned) if status else False,
            "dx": float(status.dx) if status else 0.0,
            "dy": float(status.dy) if status else 0.0,
            "stable_frames": int(status.stable_frames) if status else 0,
            "mode": "B" if getattr(self.aligner, "_body_recenter_active", False) else "H",
            "lag": float(getattr(self, "_diag_last_cmd_age_s", 0.0)),
            "interval": float(getattr(self, "_diag_move_interval_s", 0.0)),
            "settle": float(getattr(self, "_diag_settle_remaining_s", 0.0)),
            "move_interval": float(getattr(self.aligner, "_diag_move_interval_s", 0.0)),
            "last_cmd_age": float(getattr(self.aligner, "_diag_last_cmd_age_s", 0.0)),
            "ema_dx": float(getattr(self.aligner, "_ema_dx", 0.0)),
            "ema_dy": float(getattr(self.aligner, "_ema_dy", 0.0)),
            "ema_dy_rate": float(getattr(self.aligner, "_ema_dy_rate", 0.0)),
            "body_yaw": float(getattr(self.aligner, "_body_yaw", 0.0)),
        }
        try:
            self._track_log_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._track_log_fp.flush()
        except Exception:
            pass

    @staticmethod
    def parse_email_recipient_input(text: str) -> Optional[str]:
        t = (text or "").strip()
        if not t:
            return None
        if t.lower() in {"skip", "none", "no", "n"}:
            return None
        # Simple practical email validation
        if re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t):
            return t
        return None

    def _prompt_email_recipient(self) -> Optional[str]:
        # If sender is not configured, skip prompt entirely.
        if not self._email_sender.enabled:
            return None

        default_recipient = (self.cfg.email.recipient or "").strip()

        # GUI-first prompt: try a small modal input box so user can type per shot.
        try:
            import tkinter as tk
            from tkinter import simpledialog, messagebox

            root = tk.Tk()
            root.withdraw()
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass

            prompt = (
                "Enter recipient email for this photo.\n"
                "Type 'skip' to save locally only."
            )
            raw = simpledialog.askstring(
                "ReachyCheese Email",
                prompt,
                initialvalue=default_recipient,
                parent=root,
            )

            # Dialog closed/cancelled -> treat as skip local save.
            if raw is None:
                return None

            raw = (raw or "").strip()
            if not raw:
                return default_recipient or None

            parsed = self.parse_email_recipient_input(raw)
            if parsed is None and raw.lower() not in {"skip", "none", "no", "n", ""}:
                try:
                    messagebox.showwarning(
                        "Invalid email",
                        "Email format looks invalid. This capture will save locally only.",
                        parent=root,
                    )
                except Exception:
                    pass
                return None
            return parsed
        except Exception:
            # Fallback for headless/no-tk environments: terminal prompt.
            if default_recipient:
                prompt = (
                    f"📧 Enter recipient email for this photo "
                    f"(Enter=use {default_recipient}, type 'skip' to local-save only): "
                )
            else:
                prompt = "📧 Enter recipient email for this photo (or type 'skip' for local-save only): "

            try:
                raw = input(prompt)
            except EOFError:
                return default_recipient or None

            raw = (raw or "").strip()
            if not raw:
                return default_recipient or None

            parsed = self.parse_email_recipient_input(raw)
            if parsed is None and raw.lower() not in {"skip", "none", "no", "n", ""}:
                print("⚠️ Invalid email format. This capture will save locally only.")
                return None
            return parsed

    def _run_email_prompt_once(self) -> None:
        self._pending_email_recipient = self._prompt_email_recipient()

    @staticmethod
    def _is_capture_phrase(text: str) -> bool:
        normalized = ReachyCheeseApp._normalize_text(text)
        phrases = (
            "cheese",
            "cheeze",
            "take photo",
            "take picture",
            "take a photo",
            "take a picture",
            "photo",
            "picture",
        )
        if any(p in normalized for p in phrases):
            return True
        return ReachyCheeseApp._fuzzy_has_word(normalized, {"cheese", "photo", "picture"})

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _fuzzy_has_word(text: str, target_words: set[str], cutoff: float = 0.82) -> bool:
        tokens = text.split()
        for token in tokens:
            if token in target_words:
                return True
            if difflib.get_close_matches(token, list(target_words), n=1, cutoff=cutoff):
                return True
        return False

    @staticmethod
    def _is_wake_phrase(text: str, wake_word: str) -> bool:
        normalized = ReachyCheeseApp._normalize_text(text)
        aliases = {wake_word.lower(), "reachy", "ricky", "richie", "reaching"}
        if any(alias in normalized for alias in aliases):
            return True
        return ReachyCheeseApp._fuzzy_has_word(normalized, aliases, cutoff=0.78)

    @staticmethod
    def _should_disable_asr_on_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        fatal_markers = (
            "paerrorcode -9999",
            "unanticipated host error",
            "host api not found",
            "invalid input device",
            "no default input device",
            "error querying device",
        )
        return any(marker in msg for marker in fatal_markers)

    def _start_listener(self) -> None:
        if getattr(self.cfg, "asr_mode", "disable") != "vad":
            if self.cfg.debug:
                print("🎤 ASR listener disabled (asr_mode=disable)")
            return
        if self._listener_running:
            return
        self._listener_running = True

        def loop():
            while self._listener_running:
                try:
                    heard = self.voice.listen_once()
                    if heard:
                        self._asr_queue.put(heard)
                        if self.cfg.debug:
                            print(f"🎤 Heard: {heard}")
                except Exception as exc:
                    if self.cfg.debug:
                        print(f"⚠️ ASR error: {exc}")
                    if self._should_disable_asr_on_error(exc):
                        print("⚠️ ASR disabled due to audio host/device error; continuing without mic listener.")
                        self._listener_running = False
                        break
                    time.sleep(0.2)

        self._listener_thread = threading.Thread(target=loop, daemon=True)
        self._listener_thread.start()

    def _stop_listener(self) -> None:
        self._listener_running = False
        if self._listener_thread:
            self._listener_thread.join(timeout=1.5)
            self._listener_thread = None

    def _enter_sleep(self, runtime: Optional[RobotRuntime] = None) -> None:
        self.state = RCState.SLEEP
        self.aligner.reset()
        self._hint = "Say 'Reachy' to wake"
        self._countdown_overlay = ""
        if runtime is not None:
            try:
                runtime.goto_body_yaw(0.0, duration=0.55)
                runtime.reset_head(duration=0.45)
                runtime.set_antennas(0.0, 0.0, duration=0.35)
            except Exception:
                pass

    def _enter_tracking(self) -> None:
        self.state = RCState.TRACKING
        self.aligner.reset()
        self._hint = "Tracking largest face..."

    def _start_countdown(self, manual_trigger: bool = False) -> None:
        # Prompt recipient right before capture cycle so user can change per-person.
        self._run_email_prompt_once()
        self.state = RCState.COUNTDOWN
        self._countdown_started_at = time.time()
        self._countdown_index = 0
        self._countdown_overlay = "READY"
        self._countdown_antenna_phase = False
        self._countdown_started_by_manual = bool(manual_trigger)
        self._countdown_allow_auto_cancel = False
        self._countdown_cancel_grace_s = 0.6 if self._countdown_started_by_manual else 0.0
        self._countdown_outlier_streak = 0
        self._hint = "Countdown running"

    def _capture(self, runtime: Optional[RobotRuntime] = None) -> None:
        if self._last_frame is None:
            self.voice.speak_async("Sorry, I cannot get camera frame.")
            self._enter_tracking()
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.cfg.save_dir / f"IMG_{timestamp}.jpg"
        ok = cv2.imwrite(str(out_path), self._last_frame)
        if ok:
            self._last_saved_path = str(out_path)
            print(f"📸 Saved: {out_path}")
            email_msg = ""
            recipient = self._pending_email_recipient
            if self._email_sender.enabled and recipient:
                sent, err = self._email_sender.send_photo(out_path, recipient_override=recipient)
                if sent:
                    print(f"✉️ Email sent: {recipient}")
                    email_msg = " and emailed"
                else:
                    print(f"⚠️ Email send failed: {err}")
                    email_msg = ". Email failed"
            elif self._email_sender.enabled and not recipient:
                print("📭 Email skipped for this capture")
            self._pending_email_recipient = None
            self.voice.speak_async(f"Photo saved{email_msg}.")
            rt = runtime if runtime is not None else self._runtime
            self._enter_sleep(rt)
        else:
            print("❌ Failed to save photo")
            self.voice.speak_async("Failed to save photo.")
            self._enter_tracking()

    def _drain_asr(self) -> None:
        while not self._asr_queue.empty():
            text = self._asr_queue.get_nowait().lower()
            if self.cfg.debug and text:
                print(f"📝 ASR: {text}")
            if self.state == RCState.SLEEP and self._is_wake_phrase(text, self.cfg.wake_word):
                self.voice.speak_async("Hi. I am awake.")
                self._enter_tracking()
                continue
            if self.state == RCState.ARMED and self._is_capture_phrase(text):
                self._start_countdown(manual_trigger=False)
                continue

    def _drain_ui_events(self) -> bool:
        should_quit = False
        while not self._event_queue.empty():
            ev = self._event_queue.get_nowait()
            if ev == "manual_wake":
                self.voice.speak_async("Manual wake.")
                self._enter_tracking()
            elif ev == "manual_capture":
                if self.state in (RCState.TRACKING, RCState.ARMED):
                    self._start_countdown(manual_trigger=True)
            elif ev == "manual_cancel":
                if self.state == RCState.COUNTDOWN:
                    self.voice.speak_async("Countdown cancelled.")
                    self._enter_tracking()
            elif ev == "manual_sleep":
                self.voice.speak_async("Going to sleep.")
                self._enter_sleep(self._runtime)
            elif ev == "quit":
                should_quit = True
        return should_quit

    def _countdown_track_status(self, runtime: RobotRuntime, frame) -> Optional[FaceTrackStatus]:
        # Manual Take Photo countdown should keep robot still: no head/body alignment updates.
        # Voice-triggered countdown keeps low-frequency soft tracking behavior.
        if getattr(self, "_countdown_started_by_manual", False):
            return None
        return self.aligner.update(runtime, frame, soft=True)

    def _update_countdown(self, status: Optional[FaceTrackStatus], runtime: RobotRuntime) -> None:
        now = time.time()
        elapsed = now - self._countdown_started_at

        # Manual button-triggered countdown is allowed a wider deviation window
        # than voice-triggered capture, to reduce accidental cancellation.
        dx_limit = 220 if getattr(self, "_countdown_started_by_manual", False) else 140
        dy_limit = 160 if getattr(self, "_countdown_started_by_manual", False) else 110

        # Keep low-frequency face lock during countdown.
        # Auto-cancel is disabled by default for a forced countdown capture flow.
        if getattr(self, "_countdown_allow_auto_cancel", False):
            # For manual button trigger, we allow a short grace period and require
            # multiple consecutive outlier frames before cancelling.
            outlier = (not status) or (not status.has_face) or abs(status.dx) > dx_limit or abs(status.dy) > dy_limit
            if elapsed < getattr(self, "_countdown_cancel_grace_s", 0.0):
                outlier = False

            if outlier:
                self._countdown_outlier_streak = getattr(self, "_countdown_outlier_streak", 0) + 1
            else:
                self._countdown_outlier_streak = 0

            if self._countdown_outlier_streak >= getattr(self, "_countdown_cancel_streak_need", 4):
                self.voice.speak_async("Please look at me. Countdown cancelled.")
                self._enter_tracking()
                return

        while self._countdown_index < len(self._countdown_lines):
            t_mark, line = self._countdown_lines[self._countdown_index]
            if elapsed < t_mark:
                break
            self.voice.speak_async(line)

            if line.strip().lower() in self._countdown_antenna_beats:
                amp = 0.55
                if self._countdown_antenna_phase:
                    left, right = amp, -amp
                else:
                    left, right = -amp, amp
                try:
                    runtime.set_antennas(left, right, duration=0.22)
                except Exception:
                    pass
                self._countdown_antenna_phase = not self._countdown_antenna_phase

            self._countdown_index += 1

        remaining = max(0, int(4 - elapsed))
        self._countdown_overlay = f"CAPTURE IN {remaining}" if remaining > 0 else "CHEESE"
        if elapsed >= 3.4:
            self._countdown_overlay = ""
            self._capture(runtime)

    def run(self) -> None:
        if not self.gui.available:
            raise RuntimeError("GUI initialization failed.")

        self.cfg.save_dir.mkdir(parents=True, exist_ok=True)
        self._start_listener()
        self._open_track_log()

        if self.cfg.camera_source == "reachy":
            runtime: RobotRuntime = ReachyRuntime()
        else:
            runtime = WebcamRuntime(self.cfg.camera_index)

        print("🤖 ReachyCheese started")
        print(f"🎥 Camera source: {self.cfg.camera_source}")
        print(f"📁 Save dir: {self.cfg.save_dir}")

        try:
            try:
                runtime.__enter__()
                self._runtime = runtime
            except Exception as exc:
                if self.cfg.camera_source == "reachy":
                    print(f"⚠️ Reachy runtime init failed: {exc}")
                    print("↪ Falling back to webcam camera source.")
                    runtime = WebcamRuntime(self.cfg.camera_index)
                    runtime.__enter__()
                    self._runtime = runtime
                else:
                    raise

            try:
                runtime.set_automatic_body_yaw(False)
                runtime.reset_head(duration=0.5)
                frame_interval = 1.0 / max(self.cfg.preview_fps, 1.0)

                while self.gui.is_running():
                    tick = time.time()
                    if self._drain_ui_events():
                        break
                    self._drain_asr()

                    frame = runtime.get_frame()
                    if frame is None:
                        time.sleep(0.02)
                        continue
                    self._last_frame = frame.copy()
                    status = None

                    if self.state == RCState.TRACKING:
                        status = self.aligner.update(runtime, frame, soft=False)
                        if not status.has_face:
                            self._hint = "No face detected, please look at me"
                        elif status.aligned:
                            self.state = RCState.ARMED
                            self._armed_since = time.time()
                            self._hint = "Aligned. Say 'cheese' to take photo"
                            self.voice.speak_async("Look at me. Hold still.")
                        else:
                            self._hint = "Aligning..."

                    elif self.state == RCState.ARMED:
                        status = self.aligner.update(runtime, frame, soft=True)
                        if not status.has_face:
                            self.state = RCState.TRACKING
                            self._hint = "Face lost. Tracking again."
                        elif time.time() - self._armed_since > self.cfg.command_timeout_s:
                            self.voice.speak_async("Timeout. Back to sleep.")
                            self._enter_sleep(runtime)
                        else:
                            self._hint = "Say 'cheese' / 'take photo' / 'take picture'"

                    elif self.state == RCState.COUNTDOWN:
                        status = self._countdown_track_status(runtime, frame)
                        self._update_countdown(status, runtime)

                    diag_line = ""
                    if status and status.has_face:
                        mode = "B" if self.aligner._body_recenter_active else "H"
                        diag_line = (
                            f"mode={mode} dy={status.dy:+.0f} lag={self._diag_last_cmd_age_s:0.2f}s "
                            f"int={self._diag_move_interval_s:0.2f}s settle={self._diag_settle_remaining_s:0.2f}s"
                        )

                    self.gui.draw(
                        frame_bgr=frame,
                        state=self.state,
                        hint=self._hint,
                        status=status,
                        countdown_text=self._countdown_overlay,
                        last_saved=self._last_saved_path,
                        diag_line=diag_line,
                    )
                    self._append_track_log(status)

                    elapsed = time.time() - tick
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
            finally:
                runtime.__exit__(None, None, None)
                self._runtime = None
        finally:
            self._stop_listener()
            self._close_track_log()
            self.voice.close()
            self.gui.close()


def _load_json_config(config_path: Optional[str]) -> dict:
    if not config_path:
        return {}
    p = Path(os.path.expanduser(config_path))
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config file root must be a JSON object")
    return data


def _cfg_value(cli_value, config_dict: dict, key: str):
    # argparse uses None for absent optional args when default=None;
    # for booleans with store_true/store_false we avoid this helper.
    if cli_value is not None:
        return cli_value
    return config_dict.get(key)


def parse_args(argv: Optional[list[str]] = None) -> RCConfig:
    parser = argparse.ArgumentParser(description="ReachyCheese offline voice photo app")
    parser.add_argument("--config", default=None, help="Path to JSON config file")
    parser.add_argument("--preview-width", type=int, default=None)
    parser.add_argument("--preview-height", type=int, default=None)
    parser.add_argument("--preview-fps", type=float, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--wake-word", default=None)
    parser.add_argument("--asr-mode", default=None, choices=["disable", "vad"])
    parser.add_argument("--asr-model", default=None, choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--vad-silence", type=float, default=None)
    parser.add_argument("--vad-aggressive", type=int, default=None, choices=[0, 1, 2, 3])
    parser.add_argument("--piper-model", default=None)
    parser.add_argument("--piper-config", default=None)
    parser.add_argument("--speaker", type=int, default=None)
    parser.add_argument("--camera-source", default=None, choices=["reachy", "webcam"])
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--controller", default=None, choices=["v1", "v2"], help="Tracking controller version")
    parser.add_argument("--track-log", default=None, help="Path to JSONL tracking log for offline replay tuning")
    parser.add_argument("--debug", action="store_true")

    # Optional SMTP email delivery after successful capture
    parser.add_argument("--email-to", default=None)
    parser.add_argument("--email-from", default=None)
    parser.add_argument("--smtp-host", default=None)
    parser.add_argument("--smtp-port", type=int, default=None)
    parser.add_argument("--smtp-user", default=None)
    parser.add_argument("--smtp-pass", default=None)
    parser.add_argument("--smtp-encryption", default=None, choices=["start-tls", "tls", "none"])
    parser.add_argument("--email-subject", default=None)
    parser.add_argument("--email-body", default=None)

    args = parser.parse_args(argv)
    file_cfg = _load_json_config(args.config)
    email_cfg = file_cfg.get("email", {}) if isinstance(file_cfg.get("email", {}), dict) else {}

    preview_width = _cfg_value(args.preview_width, file_cfg, "preview_width") or 640
    preview_height = _cfg_value(args.preview_height, file_cfg, "preview_height") or 480
    preview_fps = _cfg_value(args.preview_fps, file_cfg, "preview_fps") or 20.0
    save_dir = _cfg_value(args.save_dir, file_cfg, "save_dir") or str(Path.home() / "Pictures" / "ReachyMiniPhoto")
    wake_word = (_cfg_value(args.wake_word, file_cfg, "wake_word") or "reachy").strip().lower()
    asr_mode = _cfg_value(args.asr_mode, file_cfg, "asr_mode")
    if asr_mode is None:
        asr_mode = "disable"
    asr_mode = str(asr_mode).strip().lower()
    if asr_mode not in {"disable", "vad"}:
        asr_mode = "disable"
    asr_model = _cfg_value(args.asr_model, file_cfg, "asr_model") or "base"
    vad_silence = _cfg_value(args.vad_silence, file_cfg, "vad_silence")
    vad_silence = 0.7 if vad_silence is None else float(vad_silence)
    vad_aggressive = _cfg_value(args.vad_aggressive, file_cfg, "vad_aggressive")
    vad_aggressive = 1 if vad_aggressive is None else int(vad_aggressive)
    piper_model = _cfg_value(args.piper_model, file_cfg, "piper_model") or "models/en-us-blizzard_lessac-medium.onnx"
    piper_config = _cfg_value(args.piper_config, file_cfg, "piper_config")
    speaker = _cfg_value(args.speaker, file_cfg, "speaker")
    speaker = 0 if speaker is None else int(speaker)
    camera_source = _cfg_value(args.camera_source, file_cfg, "camera_source") or "reachy"
    camera_index = _cfg_value(args.camera_index, file_cfg, "camera_index")
    camera_index = 0 if camera_index is None else int(camera_index)
    controller = _cfg_value(args.controller, file_cfg, "controller") or "v2"
    track_log_raw = _cfg_value(args.track_log, file_cfg, "track_log")

    email_to = _cfg_value(args.email_to, email_cfg, "to")
    if email_to is None:
        email_to = os.environ.get("REACHY_EMAIL_TO", "")
    email_from = _cfg_value(args.email_from, email_cfg, "from")
    if email_from is None:
        email_from = os.environ.get("REACHY_EMAIL_FROM", "")
    smtp_host = _cfg_value(args.smtp_host, email_cfg, "smtp_host")
    if smtp_host is None:
        smtp_host = os.environ.get("REACHY_SMTP_HOST", "smtp.gmail.com")
    smtp_port = _cfg_value(args.smtp_port, email_cfg, "smtp_port")
    if smtp_port is None:
        smtp_port = int(os.environ.get("REACHY_SMTP_PORT", "587"))
    smtp_user = _cfg_value(args.smtp_user, email_cfg, "smtp_user")
    if smtp_user is None:
        smtp_user = os.environ.get("REACHY_SMTP_USER", "")
    smtp_pass = _cfg_value(args.smtp_pass, email_cfg, "smtp_pass")
    if smtp_pass is None:
        smtp_pass = os.environ.get("REACHY_SMTP_PASS", "")
    smtp_encryption = _cfg_value(args.smtp_encryption, email_cfg, "smtp_encryption")
    if smtp_encryption is None:
        smtp_encryption = os.environ.get("REACHY_SMTP_ENCRYPTION", "start-tls")
    email_subject = _cfg_value(args.email_subject, email_cfg, "subject")
    if email_subject is None:
        email_subject = os.environ.get("REACHY_EMAIL_SUBJECT", "Reachy camera photo")
    email_body = _cfg_value(args.email_body, email_cfg, "body")
    if email_body is None:
        email_body = os.environ.get("REACHY_EMAIL_BODY", "Hi,\n\nAttached is the latest Reachy camera photo.\n\n- ReachyCheese")

    return RCConfig(
        preview_width=int(preview_width),
        preview_height=int(preview_height),
        preview_fps=float(preview_fps),
        save_dir=Path(os.path.expanduser(save_dir)),
        wake_word=wake_word,
        asr_mode=str(asr_mode),
        asr_model=asr_model,
        vad_silence=vad_silence,
        vad_aggressive=vad_aggressive,
        piper_model=piper_model,
        piper_config=piper_config,
        speaker_id=speaker,
        camera_source=camera_source,
        camera_index=camera_index,
        controller=str(controller),
        debug=bool(args.debug),
        track_log_path=Path(os.path.expanduser(str(track_log_raw))).resolve() if track_log_raw else None,
        email=EmailConfig(
            recipient=str(email_to or ""),
            sender=str(email_from or ""),
            smtp_host=str(smtp_host),
            smtp_port=int(smtp_port),
            smtp_username=str(smtp_user or ""),
            smtp_password=str(smtp_pass or ""),
            smtp_encryption=str(smtp_encryption),
            subject=str(email_subject),
            body=str(email_body),
        ),
    )


def main() -> None:
    cfg = parse_args()
    app = ReachyCheeseApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
