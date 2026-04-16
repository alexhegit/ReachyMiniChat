#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ReplayResult:
    hit_step: Optional[int]
    final_abs_dy: float
    score: float


def load_track_log(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict) or not row.get("has_face", False):
                continue
            try:
                row["dy"] = float(row.get("dy"))
            except Exception:
                continue
            rows.append(row)
    if len(rows) < 10:
        raise ValueError(f"Not enough usable rows in log: {len(rows)}")
    return rows


def replay_dy_sequence(
    dy_seq: List[float],
    params: Dict[str, float],
    gain: float,
    drag: float,
    delay_steps: int,
    target_abs_dy: float,
) -> ReplayResult:
    cmd_q = [0.0 for _ in range(max(1, delay_steps))]
    virtual_dy = float(dy_seq[0])
    hit_step: Optional[int] = None
    integral = 0.0

    for i, measured_dy in enumerate(dy_seq):
        # keep replay anchored to real trace while allowing candidate dynamics
        dy = 0.55 * float(measured_dy) + 0.45 * float(virtual_dy)
        abs_dy = abs(dy)

        if abs_dy >= params["_pitch_turbo_dy"]:
            kp_scale = params["_pitch_turbo_scale"]
        elif abs_dy >= params["_pitch_recenter_dy"]:
            kp_scale = 1.25
        else:
            kp_scale = 1.0

        kp_base = params["_pitch_kp_far"] if abs_dy >= params["_pitch_recenter_dy"] else params["_pitch_kp_near"]
        kp = kp_base * kp_scale
        kd = params["_pitch_kd"] * kp_scale

        err = params["_pitch_y_sign"] * dy
        integral += err * 0.1
        integral = float(np.clip(integral, -params["_pitch_i_clamp"], params["_pitch_i_clamp"]))

        # approximate dy rate from measured sequence
        prev = dy_seq[i - 1] if i > 0 else dy_seq[i]
        dy_rate = (dy - float(prev)) / 0.1

        cmd = kp * err + params["_pitch_ki"] * integral + kd * params["_pitch_y_sign"] * dy_rate
        cmd = float(np.clip(cmd, -params["_pitch_max_delta"], params["_pitch_max_delta"]))

        cmd_q.append(cmd)
        delayed = cmd_q.pop(0)

        # virtual plant: commanded pitch reduces dy
        virtual_dy = (virtual_dy - gain * delayed) * drag

        if hit_step is None and abs(virtual_dy) <= target_abs_dy:
            hit_step = i + 1

    final_abs = abs(float(virtual_dy))
    miss_penalty = 800.0 if hit_step is None else float(hit_step)
    score = miss_penalty + final_abs * 0.7
    return ReplayResult(hit_step=hit_step, final_abs_dy=final_abs, score=score)


def format_patch_block(params: Dict[str, float]) -> str:
    ordered = [
        "_pitch_kp_near",
        "_pitch_kp_far",
        "_pitch_ki",
        "_pitch_kd",
        "_pitch_i_clamp",
        "_pitch_max_delta",
        "_pitch_recenter_dy",
        "_pitch_turbo_dy",
        "_pitch_turbo_scale",
        "_pitch_y_sign",
    ]
    lines = ["# Paste into FaceAlignerV2.__init__ (direct pitch Y tuning)"]
    for k in ordered:
        v = params[k]
        if isinstance(v, float):
            lines.append(f"self.{k} = {v:.6g}")
        else:
            lines.append(f"self.{k} = {v}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune ReachyCheese direct-pitch Y controller from track log JSONL")
    ap.add_argument("--log", required=True, help="Path to --track-log JSONL produced by ReachyCheese")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    ap.add_argument("--target", type=float, default=50.0, help="Target |dy| threshold")
    ap.add_argument("--gain", type=float, default=65.0, help="Virtual pitch plant gain")
    ap.add_argument("--drag", type=float, default=0.992, help="Virtual pitch plant drag")
    ap.add_argument("--delay", type=int, default=7, help="Command delay steps")
    args = ap.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    rows = load_track_log(log_path)
    dy_seq = [float(r["dy"]) for r in rows]

    fixed = {
        "_pitch_i_clamp": 0.22,
        "_pitch_max_delta": 0.09,
        "_pitch_y_sign": 1.0,
    }

    grid = {
        "_pitch_kp_near": [0.0016, 0.0019, 0.0022, 0.0026],
        "_pitch_kp_far": [0.0026, 0.0032, 0.0038, 0.0044],
        "_pitch_ki": [0.00004, 0.00008, 0.00012],
        "_pitch_kd": [0.00035, 0.0006, 0.0009],
        "_pitch_recenter_dy": [100.0, 120.0, 140.0],
        "_pitch_turbo_dy": [210.0, 240.0, 270.0],
        "_pitch_turbo_scale": [1.3, 1.6, 1.9],
    }

    baseline_params = {
        **fixed,
        "_pitch_kp_near": 0.0019,
        "_pitch_kp_far": 0.0032,
        "_pitch_ki": 0.00008,
        "_pitch_kd": 0.0006,
        "_pitch_recenter_dy": 120.0,
        "_pitch_turbo_dy": 240.0,
        "_pitch_turbo_scale": 1.6,
    }

    baseline = replay_dy_sequence(
        dy_seq,
        baseline_params,
        gain=args.gain,
        drag=args.drag,
        delay_steps=args.delay,
        target_abs_dy=args.target,
    )

    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    all_results = []

    for combo in product(*vals):
        p = dict(fixed)
        p.update(dict(zip(keys, combo)))
        out = replay_dy_sequence(
            dy_seq,
            p,
            gain=args.gain,
            drag=args.drag,
            delay_steps=args.delay,
            target_abs_dy=args.target,
        )
        all_results.append(
            {
                "params": p,
                "hit_step": out.hit_step,
                "final_abs_dy": round(out.final_abs_dy, 3),
                "score": round(out.score, 3),
            }
        )

    all_results.sort(key=lambda x: x["score"])
    best = all_results[0]

    report = {
        "log": str(log_path),
        "rows": len(rows),
        "target_abs_dy": args.target,
        "baseline": {
            "hit_step": baseline.hit_step,
            "final_abs_dy": round(baseline.final_abs_dy, 3),
            "score": round(baseline.score, 3),
        },
        "best": best,
        "top5": all_results[:5],
        "patch_block": format_patch_block(best["params"]),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
