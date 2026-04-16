import unittest
from pathlib import Path
from unittest.mock import patch

from ReachyCheese import FaceTrackStatus, ReachyCheeseApp


class _DummyRuntime:
    def __init__(self):
        self.antenna_calls = []
        self.body_calls = []
        self.head_resets = []

    def set_antennas(self, left: float, right: float, duration: float = 0.2):
        self.antenna_calls.append((left, right, duration))

    def goto_body_yaw(self, yaw: float, duration: float = 0.35):
        self.body_calls.append((yaw, duration))

    def reset_head(self, duration: float = 0.2):
        self.head_resets.append(duration)


class CountdownAntennaFeatureTests(unittest.TestCase):
    def _mk_status(self, has_face: bool = True, dx: float = 0.0, dy: float = 0.0):
        return FaceTrackStatus(
            has_face=has_face,
            aligned=True,
            bbox=(0, 0, 10, 10) if has_face else None,
            center=(5, 5) if has_face else None,
            dx=dx,
            dy=dy,
            stable_frames=12 if has_face else 0,
        )

    def _mk_app(self):
        app = ReachyCheeseApp.__new__(ReachyCheeseApp)
        app._countdown_started_at = 100.0
        app._countdown_index = 0
        app._countdown_lines = [
            (0.0, "Look at me. Hold still... Ready?"),
            (0.8, "One"),
            (1.6, "Two"),
            (2.4, "Three"),
            (3.2, "Cheese"),
        ]
        app._countdown_overlay = ""
        app._hint = ""
        app.voice = type("Voice", (), {"speak_async": lambda self, text: None})()
        app._enter_tracking = lambda: None
        app._capture = lambda: None
        app.aligner = type("Aligner", (), {"reset": lambda self: None})()
        app._countdown_antenna_beats = {"one", "two", "three", "cheese"}
        app._countdown_antenna_phase = False
        app._countdown_started_by_manual = False
        app._countdown_allow_auto_cancel = False
        app._countdown_cancel_grace_s = 0.0
        app._countdown_outlier_streak = 0
        app._countdown_cancel_streak_need = 4
        return app

    def test_countdown_one_two_three_cheese_swings_antennas(self):
        app = self._mk_app()
        status = self._mk_status(has_face=True, dx=0.0, dy=0.0)
        runtime = _DummyRuntime()

        with patch("ReachyCheese.time.time", return_value=103.25):
            ReachyCheeseApp._update_countdown(app, status, runtime)

        # Expected 4 swings on One/Two/Three/Cheese (not on the initial ready sentence)
        self.assertEqual(len(runtime.antenna_calls), 4)

    def test_manual_take_photo_countdown_has_wider_cancel_tolerance(self):
        app = self._mk_app()
        app._countdown_started_by_manual = True
        app._countdown_allow_auto_cancel = True
        app._countdown_cancel_grace_s = 0.6
        app._countdown_outlier_streak = 0
        app._countdown_cancel_streak_need = 4
        status = self._mk_status(has_face=True, dx=170.0, dy=130.0)
        runtime = _DummyRuntime()

        with patch("ReachyCheese.time.time", return_value=100.1):
            ReachyCheeseApp._update_countdown(app, status, runtime)

        # Should NOT be cancelled immediately for manual trigger
        self.assertEqual(app._countdown_index, 1)
        self.assertEqual(app._countdown_outlier_streak, 0)

    def test_manual_take_photo_cancels_after_consecutive_outliers_post_grace(self):
        app = self._mk_app()
        app._countdown_started_by_manual = True
        app._countdown_allow_auto_cancel = True
        app._countdown_cancel_grace_s = 0.0
        app._countdown_outlier_streak = 0
        app._countdown_cancel_streak_need = 4

        status = self._mk_status(has_face=True, dx=240.0, dy=170.0)
        runtime = _DummyRuntime()
        marker = {"entered_tracking": False}

        def _enter_tracking():
            marker["entered_tracking"] = True

        app._enter_tracking = _enter_tracking

        for i in range(4):
            with patch("ReachyCheese.time.time", return_value=100.8 + i * 0.05):
                ReachyCheeseApp._update_countdown(app, status, runtime)

        self.assertTrue(marker["entered_tracking"])

    def test_voice_countdown_still_cancels_on_large_deviation(self):
        app = self._mk_app()
        app._countdown_started_by_manual = False
        app._countdown_allow_auto_cancel = True
        app._countdown_cancel_grace_s = 0.0
        app._countdown_outlier_streak = 0
        app._countdown_cancel_streak_need = 1
        status = self._mk_status(has_face=True, dx=170.0, dy=130.0)
        runtime = _DummyRuntime()

        marker = {"entered_tracking": False}

        def _enter_tracking():
            marker["entered_tracking"] = True

        app._enter_tracking = _enter_tracking

        with patch("ReachyCheese.time.time", return_value=100.1):
            ReachyCheeseApp._update_countdown(app, status, runtime)

        self.assertTrue(marker["entered_tracking"])

    def test_forced_countdown_does_not_auto_cancel_on_large_deviation(self):
        app = self._mk_app()
        app._countdown_started_by_manual = True
        app._countdown_allow_auto_cancel = False
        status = self._mk_status(has_face=True, dx=260.0, dy=200.0)
        runtime = _DummyRuntime()

        marker = {"entered_tracking": False}

        def _enter_tracking():
            marker["entered_tracking"] = True

        app._enter_tracking = _enter_tracking

        with patch("ReachyCheese.time.time", return_value=100.2):
            ReachyCheeseApp._update_countdown(app, status, runtime)

        self.assertFalse(marker["entered_tracking"])

    def test_enter_sleep_resets_robot_pose_and_antennas(self):
        app = self._mk_app()
        runtime = _DummyRuntime()

        ReachyCheeseApp._enter_sleep(app, runtime)

        self.assertEqual(len(runtime.body_calls), 1)
        self.assertEqual(runtime.body_calls[0][0], 0.0)
        self.assertEqual(len(runtime.head_resets), 1)
        self.assertEqual(len(runtime.antenna_calls), 1)

    def test_capture_uses_explicit_runtime_to_enter_sleep(self):
        app = self._mk_app()
        runtime = _DummyRuntime()
        app._last_frame = object()
        app.cfg = type("Cfg", (), {"save_dir": Path("/tmp")})()
        app._pending_email_recipient = None
        app._email_sender = type("Sender", (), {"enabled": False})()
        app._runtime = None

        marker = {"rt": None}

        def _enter_sleep(rt):
            marker["rt"] = rt

        app._enter_sleep = _enter_sleep

        with patch("ReachyCheese.datetime") as dt, \
             patch("ReachyCheese.cv2.imwrite", return_value=True):
            dt.now.return_value.strftime.return_value = "20260101_000000"
            ReachyCheeseApp._capture(app, runtime)

        self.assertIs(marker["rt"], runtime)

    def test_capture_falls_back_to_app_runtime_when_not_passed(self):
        app = self._mk_app()
        fallback_rt = _DummyRuntime()
        app._last_frame = object()
        app.cfg = type("Cfg", (), {"save_dir": Path("/tmp")})()
        app._pending_email_recipient = None
        app._email_sender = type("Sender", (), {"enabled": False})()
        app._runtime = fallback_rt

        marker = {"rt": None}

        def _enter_sleep(rt):
            marker["rt"] = rt

        app._enter_sleep = _enter_sleep

        with patch("ReachyCheese.datetime") as dt, \
             patch("ReachyCheese.cv2.imwrite", return_value=True):
            dt.now.return_value.strftime.return_value = "20260101_000000"
            ReachyCheeseApp._capture(app)

        self.assertIs(marker["rt"], fallback_rt)

    def test_manual_countdown_skips_alignment_updates_to_keep_robot_still(self):
        app = self._mk_app()
        app._countdown_started_by_manual = True

        marker = {"called": 0}

        class _Aligner:
            def update(self, runtime, frame, soft=False):
                marker["called"] += 1
                return object()

        app.aligner = _Aligner()

        status = ReachyCheeseApp._countdown_track_status(app, _DummyRuntime(), object())
        self.assertIsNone(status)
        self.assertEqual(marker["called"], 0)

    def test_voice_countdown_keeps_soft_alignment_updates(self):
        app = self._mk_app()
        app._countdown_started_by_manual = False
        expected = object()

        marker = {"called": 0, "soft": None}

        class _Aligner:
            def update(self, runtime, frame, soft=False):
                marker["called"] += 1
                marker["soft"] = soft
                return expected

        app.aligner = _Aligner()

        status = ReachyCheeseApp._countdown_track_status(app, _DummyRuntime(), object())
        self.assertIs(status, expected)
        self.assertEqual(marker["called"], 1)
        self.assertTrue(marker["soft"])


if __name__ == "__main__":
    unittest.main()
