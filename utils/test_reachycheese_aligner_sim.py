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
        self.body_cmds = []
        self.reset_head_cmds = 0

    def look_at_image(self, x, y, duration=0.2):
        self.look_cmds.append((int(x), int(y), float(duration)))

    def goto_body_yaw(self, yaw, duration=0.35):
        self.body_cmds.append((float(yaw), float(duration)))

    def reset_head(self, duration=0.2):
        self.reset_head_cmds += 1


def main() -> int:
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
