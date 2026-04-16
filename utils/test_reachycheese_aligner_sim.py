#!/usr/bin/env python3
"""Quick test of FaceAligner logic with synthetic face boxes (no camera/hardware).

This validates state progression and motion command issuance logic:
- tracking updates produce look_at_image calls when offset is large
- stable centered detections eventually report aligned=True
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ReachyCheese import FaceAligner


class DummyTracker:
    def __init__(self, boxes):
        self._boxes = list(boxes)
        self._idx = 0

    def detect(self, frame):
        if self._idx >= len(self._boxes):
            return self._boxes[-1] if self._boxes else None
        v = self._boxes[self._idx]
        self._idx += 1
        return v


class DummyRuntime:
    def __init__(self):
        self.look_cmds = []
        self.pitch_cmds = []
        self.body_cmds = []
        self.reset_head_cmds = 0

    def look_at_image(self, x, y, duration=0.2):
        self.look_cmds.append((int(x), int(y), float(duration)))

    def set_head_pitch(self, pitch, duration=0.2):
        self.pitch_cmds.append((float(pitch), float(duration)))

    def goto_body_yaw(self, yaw, duration=0.35):
        self.body_cmds.append((float(yaw), float(duration)))

    def reset_head(self, duration=0.2):
        self.reset_head_cmds += 1


def test_vertical_force_resend_when_target_pixel_is_stable():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # centered x, strong downward correction needed (large positive dy)
    box = (260, 300, 120, 120)  # center y=360 -> dy=+120

    aligner = FaceAligner()
    aligner._tracker = DummyTracker([box, box])

    rt = DummyRuntime()

    # First update should send one look command.
    aligner._last_track_at = 0.0
    with patch("ReachyCheese.time.time", return_value=10.0):
        aligner.update(rt, frame, soft=False)

    # On direct-pitch Y mode, look_at_image keeps Y at center while vertical
    # correction is applied through set_head_pitch.
    assert rt.look_cmds, "expected first look command"
    first_y = rt.look_cmds[0][1]
    assert first_y == frame.shape[0] // 2, f"expected centered look-at Y in direct-pitch mode, got {first_y}"
    assert rt.pitch_cmds, "expected direct pitch command on first update"

    # Second update keeps almost same target pixel (dedupe path), but enough time elapsed
    # and vertical error is large, so force-resend should emit another command.
    aligner._last_track_at = 0.0
    with patch("ReachyCheese.time.time", return_value=10.8):
        aligner.update(rt, frame, soft=False)

    assert len(rt.look_cmds) >= 2, f"expected force-resend look cmd, got {len(rt.look_cmds)}"


def test_vertical_sign_flip_forces_immediate_brake_command():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # First frame: face below center (dy > 0) -> downward correction command.
    below_box = (260, 300, 120, 120)  # cy=360 -> dy=+120
    # Second frame: slight overshoot to above center (dy < 0), but similar target pixels.
    above_box = (260, 0, 120, 120)    # cy=60 -> dy=-180 (ensures |ema_dy| stays above release)

    aligner = FaceAligner()
    aligner._tracker = DummyTracker([below_box, above_box])
    aligner._force_resend_interval_s = 99.0  # disable periodic resend to isolate sign-flip behavior

    rt = DummyRuntime()

    aligner._last_track_at = 0.0
    with patch("ReachyCheese.time.time", return_value=20.0):
        aligner.update(rt, frame, soft=False)

    first_count = len(rt.look_cmds)
    assert first_count >= 1, "expected initial vertical correction command"

    aligner._last_track_at = 0.0
    with patch("ReachyCheese.time.time", return_value=20.1):
        aligner.update(rt, frame, soft=False)

    assert len(rt.look_cmds) >= first_count + 1, "expected immediate brake command on dy sign flip"
    assert len(rt.pitch_cmds) >= first_count + 1, "expected direct pitch command on dy sign flip"


def test_direct_pitch_command_is_rate_limited_and_bounded():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Use two strong below-center detections then a milder below-center detection to
    # create positive dy while dy_rate is strongly negative (approaching center quickly).
    boxes = [
        (260, 340, 120, 120),  # dy=+160
        (260, 340, 120, 120),  # dy=+160
        (260, 200, 120, 120),  # dy=+20, approaching center fast
    ]

    aligner = FaceAligner()
    aligner._tracker = DummyTracker(boxes)
    aligner._min_cmd_delta_px = 0
    aligner._last_track_at = 0.0

    rt = DummyRuntime()

    with patch("ReachyCheese.time.time", side_effect=[30.0, 30.10, 30.20]):
        aligner.update(rt, frame, soft=False)
        aligner._last_track_at = 0.0
        aligner.update(rt, frame, soft=False)
        aligner._last_track_at = 0.0
        aligner.update(rt, frame, soft=False)

    assert len(rt.look_cmds) >= 3, "expected 3 look commands"
    assert len(rt.pitch_cmds) >= 3, "expected 3 direct pitch commands"
    p1 = rt.pitch_cmds[0][0]
    p2 = rt.pitch_cmds[1][0]
    p3 = rt.pitch_cmds[2][0]
    max_delta = aligner._pitch_max_delta + 1e-6
    assert abs(p2 - p1) <= max_delta, f"pitch delta too large: {p2-p1}"
    assert abs(p3 - p2) <= max_delta, f"pitch delta too large: {p3-p2}"
    assert abs(p3) <= aligner._head_pitch_max + 1e-6, f"pitch exceeded bound: {p3}"


def test_far_vertical_error_uses_reacquire_profile():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Prime with two very large dy frames so EMA clearly enters far-reacquire branch,
    # then inspect the third command amplitude.
    far_below_box = (260, 420, 120, 120)   # cy=480 -> dy=+240
    far_below_box2 = (260, 460, 120, 120)  # cy=520 -> dy=+280
    far_below_box3 = (260, 460, 120, 120)  # keep large for stable assertion

    aligner = FaceAligner()
    aligner._tracker = DummyTracker([far_below_box, far_below_box2, far_below_box3])

    rt = DummyRuntime()

    aligner._last_track_at = 0.0
    with patch("ReachyCheese.time.time", return_value=40.0):
        aligner.update(rt, frame, soft=False)

    aligner._last_track_at = 0.0
    with patch("ReachyCheese.time.time", return_value=40.3):
        aligner.update(rt, frame, soft=False)

    aligner._last_track_at = 0.0
    with patch("ReachyCheese.time.time", return_value=40.6):
        aligner.update(rt, frame, soft=False)

    assert rt.pitch_cmds, "expected direct pitch command"
    p = abs(rt.pitch_cmds[-1][0])
    assert p <= aligner._head_pitch_max, f"pitch {p} exceeds limit"
    assert p >= 0.20, f"expected stronger far-profile pitch correction, got |pitch|={p}"


def test_large_dy_still_emits_head_y_command_during_body_recenter_stage():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Large dx keeps body-recenter mode active; large dy should still trigger Y-only head correction.
    # center x=820 -> dx=+500, center y=520 -> dy=+280
    box = (760, 460, 120, 120)

    aligner = FaceAligner()
    aligner._tracker = DummyTracker([box, box, box])

    rt = DummyRuntime()

    # First updates can trigger body recenter gating; by third we should still emit head command on Y.
    aligner._last_track_at = 0.0
    with patch("ReachyCheese.time.time", return_value=50.0):
        aligner.update(rt, frame, soft=False)

    aligner._last_track_at = 0.0
    with patch("ReachyCheese.time.time", return_value=50.4):
        aligner.update(rt, frame, soft=False)

    aligner._last_track_at = 0.0
    with patch("ReachyCheese.time.time", return_value=50.8):
        st = aligner.update(rt, frame, soft=False)

    assert aligner._body_recenter_active, "expected body recenter stage to be active"
    assert st.dy > 160, f"expected large dy, got {st.dy}"
    assert len(rt.pitch_cmds) >= 1, "expected Y direct pitch command even during body recenter"


def main() -> int:
    test_vertical_force_resend_when_target_pixel_is_stable()
    test_vertical_sign_flip_forces_immediate_brake_command()
    test_direct_pitch_command_is_rate_limited_and_bounded()
    test_far_vertical_error_uses_reacquire_profile()
    test_large_dy_still_emits_head_y_command_during_body_recenter_stage()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Sequence: off-center right first (should trigger look commands), then centered stable.
    off_center_box = (420, 140, 120, 120)  # center at x=480 (dx ~ +160)
    centered_box = (260, 180, 120, 120)    # center near (320,240)
    boxes = [off_center_box] * 8 + [centered_box] * 16

    aligner = FaceAligner()
    aligner._tracker = DummyTracker(boxes)

    rt = DummyRuntime()
    statuses = []
    for _ in range(len(boxes)):
        # Force command-interval eligibility so this logic test is deterministic
        # even when the loop runs much faster than real-time.
        aligner._last_track_at = 0.0
        st = aligner.update(rt, frame, soft=False)
        statuses.append({
            "has_face": st.has_face,
            "aligned": st.aligned,
            "stable_frames": st.stable_frames,
            "dx": round(float(st.dx), 2),
            "dy": round(float(st.dy), 2),
        })

    final_aligned = any(s["aligned"] for s in statuses)
    summary = {
        "look_commands": len(rt.look_cmds),
        "body_commands": len(rt.body_cmds),
        "reset_head_commands": rt.reset_head_cmds,
        "ever_aligned": final_aligned,
        "last_status": statuses[-1],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # expectations for logic health
    if len(rt.look_cmds) == 0:
        return 1
    if not final_aligned:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
