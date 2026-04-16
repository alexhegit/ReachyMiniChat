from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase

from utils.tune_vertical_from_log import format_patch_block, load_track_log


class TestTuneVerticalFromLog(TestCase):
    def test_load_track_log_filters_invalid_rows(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "track.jsonl"
            rows = [
                {"has_face": True, "dy": 300},
                {"has_face": False, "dy": 120},
                {"has_face": True, "dy": "bad"},
                {"has_face": True, "dy": 260},
                {"has_face": True, "dy": 210},
                {"has_face": True, "dy": 180},
                {"has_face": True, "dy": 150},
                {"has_face": True, "dy": 120},
                {"has_face": True, "dy": 95},
                {"has_face": True, "dy": 80},
                {"has_face": True, "dy": 65},
                {"has_face": True, "dy": 55},
            ]
            with open(p, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            out = load_track_log(p)
            self.assertGreaterEqual(len(out), 10)
            self.assertEqual(out[0]["dy"], 300.0)

    def test_format_patch_block_contains_assignments(self):
        params = {
            "_pitch_kp_near": 0.0019,
            "_pitch_kp_far": 0.0032,
            "_pitch_ki": 0.00008,
            "_pitch_kd": 0.0006,
            "_pitch_i_clamp": 0.22,
            "_pitch_max_delta": 0.09,
            "_pitch_recenter_dy": 120.0,
            "_pitch_turbo_dy": 240.0,
            "_pitch_turbo_scale": 1.6,
            "_pitch_y_sign": 1.0,
        }
        block = format_patch_block(params)
        self.assertIn("self._pitch_kp_near", block)
        self.assertIn("self._pitch_turbo_scale", block)
