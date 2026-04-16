from __future__ import annotations

from collections import deque
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from ReachyCheese import FaceAligner


class _DummyRuntime:
    def __init__(self):
        self.look_cmds = []
        self.pitch_cmds = []
        self.body_cmds = []

    def look_at_image(self, x, y, duration=0.2):
        self.look_cmds.append((int(x), int(y), float(duration)))

    def set_head_pitch(self, pitch, duration=0.2):
        self.pitch_cmds.append((float(pitch), float(duration)))

    def goto_body_yaw(self, yaw, duration=0.35):
        self.body_cmds.append((float(yaw), float(duration)))

    def reset_head(self, duration=0.2):
        return None


def _box_from_dxdy(dx: float, dy: float, frame_w: int = 640, frame_h: int = 480):
    w, h = 120, 120
    cx = int(frame_w // 2 + dx)
    cy = int(frame_h // 2 + dy)
    return (cx - w // 2, cy - h // 2, w, h)


def _run_vertical_recovery_sim(
    *,
    x0: float,
    y0: float,
    delay_steps: int,
    plant_gain: float,
    drag: float,
    steps: int = 180,
    dt_s: float = 0.1,
):
    aligner = FaceAligner()
    runtime = _DummyRuntime()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    dx = float(x0)
    dy = float(y0)
    cmd_queue = deque([0.0] * delay_steps, maxlen=delay_steps)
    center_y = frame.shape[0] // 2

    first_hit_step = None
    for i in range(steps):
        t = 100.0 + i * dt_s
        box = _box_from_dxdy(dx, dy)
        aligner._tracker = type("_Tracker", (), {"detect": lambda self, fr, b=box: b})()
        aligner._last_track_at = 0.0

        with patch("ReachyCheese.time.time", return_value=t):
            aligner.update(runtime, frame, soft=False)

        if runtime.pitch_cmds:
            cmd_value = float(runtime.pitch_cmds[-1][0])
        else:
            cmd_value = 0.0
            if runtime.look_cmds:
                cmd_value = float(runtime.look_cmds[-1][1] - center_y)

        cmd_queue.append(cmd_value)
        delayed_cmd = cmd_queue[0]

        # Plant model supports both control styles:
        # - direct pitch servo: dy responds strongly to pitch angle
        # - image-space look-at fallback: dy responds to target-y offset
        if runtime.pitch_cmds:
            dy = (dy - 65.0 * delayed_cmd) * 0.992
        else:
            dy = (dy - plant_gain * delayed_cmd) * drag

        if first_hit_step is None and abs(dy) <= 50.0:
            first_hit_step = i + 1

    return {
        "hit_step": first_hit_step,
        "final_dy": float(dy),
        "look_cmd_count": len(runtime.look_cmds),
        "pitch_cmd_count": len(runtime.pitch_cmds),
        "body_cmd_count": len(runtime.body_cmds),
    }


class TestFaceAlignerVerticalRecovery(TestCase):
    def test_vertical_recovery_from_dy300_under_stiff_plant_reaches_50(self):
        # Stiff + laggy vertical plant that previously got stuck just above |dy|=50.
        result = _run_vertical_recovery_sim(
            x0=0.0,
            y0=300.0,
            delay_steps=7,
            plant_gain=0.020,
            drag=0.998,
            steps=180,
            dt_s=0.1,
        )

        self.assertIsNotNone(result["hit_step"], f"never reached |dy|<=50, final={result['final_dy']:.2f}")
        self.assertLessEqual(result["hit_step"], 80, f"converged too slowly: {result['hit_step']} steps")
        self.assertGreater(result["pitch_cmd_count"], 0, "expected direct pitch commands")

    def test_vertical_recovery_during_body_recenter_mode_reaches_50(self):
        # Large horizontal error keeps body-recenter active; Y recovery must still succeed.
        result = _run_vertical_recovery_sim(
            x0=260.0,
            y0=300.0,
            delay_steps=7,
            plant_gain=0.022,
            drag=0.997,
            steps=180,
            dt_s=0.1,
        )

        self.assertGreater(result["body_cmd_count"], 0, "expected body recenter commands to be exercised")
        self.assertIsNotNone(result["hit_step"], f"never reached |dy|<=50, final={result['final_dy']:.2f}")
        self.assertLessEqual(result["hit_step"], 95, f"converged too slowly with body recenter: {result['hit_step']}")
