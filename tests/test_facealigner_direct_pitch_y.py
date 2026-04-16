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


def _run_direct_pitch_recovery(
    *,
    x0: float,
    y0: float,
    delay_steps: int,
    gain: float,
    drag: float,
    steps: int = 180,
    dt_s: float = 0.1,
):
    aligner = FaceAligner()
    aligner._use_direct_pitch_y = True

    runtime = _DummyRuntime()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    dx = float(x0)
    dy = float(y0)
    cmd_q = deque([0.0] * delay_steps, maxlen=delay_steps)

    hit_step = None
    for i in range(steps):
        t = 200.0 + i * dt_s
        box = _box_from_dxdy(dx, dy)
        aligner._tracker = type("_Tracker", (), {"detect": lambda self, fr, b=box: b})()
        aligner._last_track_at = 0.0

        with patch("ReachyCheese.time.time", return_value=t):
            aligner.update(runtime, frame, soft=False)

        pitch = runtime.pitch_cmds[-1][0] if runtime.pitch_cmds else 0.0
        cmd_q.append(float(pitch))
        delayed = cmd_q[0]

        # Virtual pitch plant: commanded pitch reduces dy with delay and drag.
        dy = (dy - gain * delayed) * drag

        if hit_step is None and abs(dy) <= 50.0:
            hit_step = i + 1

    return {
        "hit_step": hit_step,
        "final_dy": float(dy),
        "pitch_cmd_count": len(runtime.pitch_cmds),
        "look_cmd_count": len(runtime.look_cmds),
        "body_cmd_count": len(runtime.body_cmds),
    }


class TestFaceAlignerDirectPitchY(TestCase):
    def test_direct_pitch_y_recovers_from_dy300_to_within_50(self):
        result = _run_direct_pitch_recovery(
            x0=0.0,
            y0=300.0,
            delay_steps=7,
            gain=65.0,
            drag=0.992,
            steps=180,
            dt_s=0.1,
        )

        self.assertIsNotNone(result["hit_step"], f"never reached |dy|<=50, final={result['final_dy']:.2f}")
        self.assertLessEqual(result["hit_step"], 80, f"direct pitch converged too slowly: {result['hit_step']}")
        self.assertGreater(result["pitch_cmd_count"], 0, "expected direct pitch commands")

    def test_direct_pitch_still_works_while_body_recenter_active(self):
        result = _run_direct_pitch_recovery(
            x0=260.0,
            y0=300.0,
            delay_steps=7,
            gain=65.0,
            drag=0.992,
            steps=180,
            dt_s=0.1,
        )

        self.assertGreater(result["body_cmd_count"], 0, "expected body recenter branch exercised")
        self.assertIsNotNone(result["hit_step"], f"never reached |dy|<=50, final={result['final_dy']:.2f}")
        self.assertLessEqual(result["hit_step"], 95, f"direct pitch with body recenter too slow: {result['hit_step']}")
