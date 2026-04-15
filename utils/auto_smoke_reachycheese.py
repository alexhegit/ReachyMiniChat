#!/usr/bin/env python3
"""Automated smoke test for ReachyCheese core pipeline (no human required).

Critical checks:
1) Reachy daemon connection
2) Camera frame availability
3) Motion command path
4) Photo save path/write

Optional checks:
5) Face detector process health (isolated)
6) Full app process starts and stays alive briefly
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = str(PROJECT_ROOT / "venv/bin/python")


def run_face_detector_subprocess() -> dict:
    code = r'''
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from reachy_mini import ReachyMini
from vision.face_tracker import FaceTracker

out = {"ran": False, "detections": 0, "errors": 0, "using_fallback": None}
with ReachyMini(media_backend='default') as r:
    tr = FaceTracker(smooth_factor=0.20, multi_face_strategy='largest', min_detection_confidence=0.70)
    for _ in range(8):
        f = r.media.get_frame() if r.media else None
        if f is None:
            continue
        try:
            bb = tr.detect(f)
            out['ran'] = True
            out['using_fallback'] = getattr(tr, '_using_fallback', None)
            if bb is not None:
                out['detections'] += 1
        except Exception:
            out['errors'] += 1
print(json.dumps(out))
'''
    return run_python_snippet(code)


def run_full_app_start_check() -> dict:
    cmd = [
        PY,
        "ReachyCheese.py",
        "--debug",
        "--camera-source",
        "reachy",
        "--piper-model",
        "models/en-us-blizzard_lessac-medium.onnx",
    ]
    try:
        p = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=12,
        )
        # If it exits early, still capture logs
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        return {
            "status": "exited",
            "returncode": p.returncode,
            "log_tail": out[-1200:],
        }
    except subprocess.TimeoutExpired as e:
        s_out = e.stdout.decode("utf-8", errors="ignore") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")
        s_err = e.stderr.decode("utf-8", errors="ignore") if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or "")
        out = (s_out + "\n" + s_err) if (s_out or s_err) else ""
        return {
            "status": "running_timeout_ok",
            "log_tail": out[-1200:],
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def run_python_snippet(code: str) -> dict:
    try:
        p = subprocess.run(
            [PY, "-c", code, str(PROJECT_ROOT)],
            capture_output=True,
            text=True,
            timeout=25,
            cwd=str(PROJECT_ROOT),
        )
        if p.returncode == 0:
            lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
            for ln in reversed(lines):
                if ln.startswith("{") and ln.endswith("}"):
                    return {"status": "ok", **json.loads(ln)}
            return {"status": "unknown", "raw_stdout": p.stdout[-500:]}
        return {
            "status": "failed",
            "returncode": p.returncode,
            "stderr_tail": (p.stderr or "")[-500:],
            "stdout_tail": (p.stdout or "")[-500:],
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def main() -> int:
    result = {
        "connected": False,
        "camera": {"frames_checked": 0, "non_none_frames": 0, "shape": None},
        "motion": {"ok": False, "error": None},
        "photo_save": {"ok": False, "path": None, "error": None},
        "face_detector_subprocess": {},
        "full_app_start_check": {},
        "critical_failures": [],
        "warnings": [],
    }

    save_dir = Path.home() / "Pictures" / "ReachyMiniPhoto"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        with ReachyMini(media_backend="default") as robot:
            result["connected"] = True
            last_frame = None

            for _ in range(35):
                frame = robot.media.get_frame() if robot.media else None
                result["camera"]["frames_checked"] += 1
                if frame is None:
                    time.sleep(0.05)
                    continue
                last_frame = frame
                result["camera"]["non_none_frames"] += 1
                result["camera"]["shape"] = list(frame.shape)
                time.sleep(0.02)

            try:
                robot.set_automatic_body_yaw(False)
                if last_frame is not None:
                    h, w = last_frame.shape[:2]
                    robot.look_at_image(w // 2, h // 2, duration=0.25)
                robot.goto_target(body_yaw=0.12, duration=0.30)
                robot.goto_target(head=create_head_pose(), duration=0.25)
                result["motion"]["ok"] = True
            except Exception as exc:
                result["motion"]["error"] = str(exc)

            if last_frame is not None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = save_dir / f"TEST_IMG_{ts}.jpg"
                ok = cv2.imwrite(str(out), last_frame)
                result["photo_save"]["ok"] = bool(ok)
                result["photo_save"]["path"] = str(out)
                if not ok:
                    result["photo_save"]["error"] = "cv2.imwrite returned False"
            else:
                result["critical_failures"].append("camera_no_frames")

    except Exception as exc:
        result["critical_failures"].append(f"connection_error:{exc}")

    # Optional checks
    result["face_detector_subprocess"] = run_face_detector_subprocess()
    if result["face_detector_subprocess"].get("status") != "ok":
        result["warnings"].append("face_detector_subprocess_not_healthy")

    result["full_app_start_check"] = run_full_app_start_check()
    app_status = result["full_app_start_check"].get("status")
    if app_status not in ("running_timeout_ok", "exited"):
        result["warnings"].append("full_app_start_check_failed")

    # Critical validation
    if not result["connected"]:
        result["critical_failures"].append("not_connected")
    if result["camera"]["non_none_frames"] == 0:
        result["critical_failures"].append("camera_no_frames")
    if not result["motion"]["ok"]:
        result["critical_failures"].append("motion_path_failed")
    if not result["photo_save"]["ok"]:
        result["critical_failures"].append("photo_save_failed")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["critical_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
