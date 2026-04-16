#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ReachyCheese import FaceAligner  # noqa: E402


class DummyRuntime:
    def __init__(self):
        self.look_cmds = []
        self.body_cmds = []

    def look_at_image(self, x, y, duration=0.2):
        self.look_cmds.append((int(x), int(y), float(duration)))

    def goto_body_yaw(self, yaw, duration=0.35):
        self.body_cmds.append((float(yaw), float(duration)))

    def reset_head(self, duration=0.2):
        return None


@dataclass
class Scenario:
    name: str
    x0: float
    y0: float
    delay: int
    gain: float
    drag: float


@dataclass
class ScenarioResult:
    hit_step: Optional[int]
    final_dy: float
    look_cmd_count: int
    body_cmd_count: int


def box_from_dxdy(dx: float, dy: float, frame_w: int = 640, frame_h: int = 480):
    w, h = 120, 120
    cx = int(frame_w // 2 + dx)
    cy = int(frame_h // 2 + dy)
    return (cx - w // 2, cy - h // 2, w, h)


def run_sim(params: Dict[str, float], sc: Scenario, steps: int = 180, dt_s: float = 0.1) -> ScenarioResult:
    aligner = FaceAligner()
    for k, v in params.items():
        setattr(aligner, k, v)

    runtime = DummyRuntime()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    center_y = frame.shape[0] // 2

    dx = float(sc.x0)
    dy = float(sc.y0)
    q = deque([0.0] * sc.delay, maxlen=sc.delay)
    hit_step: Optional[int] = None

    for i in range(steps):
        t = 100.0 + i * dt_s
        box = box_from_dxdy(dx, dy)
        aligner._tracker = type("_Tracker", (), {"detect": lambda self, fr, b=box: b})()
        aligner._last_track_at = 0.0

        with patch("ReachyCheese.time.time", return_value=t):
            aligner.update(runtime, frame, soft=False)

        cmd_offset_y = 0.0
        if runtime.look_cmds:
            cmd_offset_y = float(runtime.look_cmds[-1][1] - center_y)
        q.append(cmd_offset_y)
        delayed_cmd = q[0]

        dy = dy - sc.gain * delayed_cmd
        dy = dy * sc.drag

        if hit_step is None and abs(dy) <= 50.0:
            hit_step = i + 1

    return ScenarioResult(
        hit_step=hit_step,
        final_dy=float(dy),
        look_cmd_count=len(runtime.look_cmds),
        body_cmd_count=len(runtime.body_cmds),
    )


def main() -> int:
    scenarios = [
        Scenario("stiff_center", x0=0.0, y0=300.0, delay=7, gain=0.020, drag=0.998),
        Scenario("body_recenter", x0=260.0, y0=300.0, delay=7, gain=0.022, drag=0.997),
        Scenario("laggy_center", x0=0.0, y0=300.0, delay=9, gain=0.024, drag=0.997),
    ]

    grid = {
        "_vertical_reacquire_dy": [120, 130, 140, 160],
        "_target_y_gain_down_far": [0.78, 0.86, 0.94],
        "_target_y_gain_up_far": [0.82, 0.90, 0.98],
        "_target_y_max_step_down_far": [118, 126, 136],
        "_target_y_max_step_up_far": [122, 130, 140],
        "_min_cmd_delta_px": [20, 22, 24],
        "_force_resend_interval_s": [0.55, 0.65, 0.72],
    }

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    best = None
    candidates = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        total_score = 0.0
        details: Dict[str, Dict] = {}
        hard_ok = True

        for sc in scenarios:
            out = run_sim(params, sc)
            details[sc.name] = asdict(out)
            hit = out.hit_step if out.hit_step is not None else 999
            total_score += hit + 0.15 * out.look_cmd_count

            if sc.name == "stiff_center" and (out.hit_step is None or out.hit_step > 165):
                hard_ok = False
            if sc.name == "body_recenter" and (out.hit_step is None or out.hit_step > 165):
                hard_ok = False

        if not hard_ok:
            continue

        row = {"score": round(total_score, 3), "params": params, "details": details}
        candidates.append(row)

    candidates.sort(key=lambda x: x["score"])
    best = candidates[0] if candidates else None

    result = {
        "search_space_size": int(np.prod([len(v) for v in values])),
        "qualified_candidates": len(candidates),
        "best": best,
        "top5": candidates[:5],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if best else 1


if __name__ == "__main__":
    raise SystemExit(main())
