import unittest

from ReachyCheese import RCConfig, ReachyCheeseApp, parse_args


class ASRModeFeatureTests(unittest.TestCase):
    def test_parse_args_defaults_asr_mode_disable(self):
        cfg = parse_args([])
        self.assertEqual(cfg.asr_mode, "disable")

    def test_parse_args_accepts_asr_mode_vad(self):
        cfg = parse_args(["--asr-mode", "vad"])
        self.assertEqual(cfg.asr_mode, "vad")

    def test_start_listener_noops_when_asr_mode_disable(self):
        app = ReachyCheeseApp.__new__(ReachyCheeseApp)
        app.cfg = RCConfig(asr_mode="disable")
        app._listener_running = False
        app._listener_thread = None
        app.voice = type("Voice", (), {"listen_once": lambda self: "wake"})()
        app._asr_queue = None

        ReachyCheeseApp._start_listener(app)

        self.assertFalse(app._listener_running)
        self.assertIsNone(app._listener_thread)


if __name__ == "__main__":
    unittest.main()
