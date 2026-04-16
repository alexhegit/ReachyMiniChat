import json
import tempfile
import unittest
from pathlib import Path

from ReachyCheese import parse_args


class ConfigFileFeatureTests(unittest.TestCase):
    def test_parse_args_reads_json_config(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "reachycheese.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "camera_source": "webcam",
                        "camera_index": 2,
                        "controller": "v1",
                        "preview_width": 800,
                        "email": {
                            "to": "person@example.com",
                            "from": "robot@example.com",
                            "smtp_host": "smtp.example.com",
                            "smtp_port": 465,
                            "smtp_encryption": "tls",
                        },
                    }
                ),
                encoding="utf-8",
            )

            cfg = parse_args(["--config", str(cfg_path)])

            self.assertEqual(cfg.camera_source, "webcam")
            self.assertEqual(cfg.camera_index, 2)
            self.assertEqual(cfg.controller, "v1")
            self.assertEqual(cfg.preview_width, 800)
            self.assertEqual(cfg.email.recipient, "person@example.com")
            self.assertEqual(cfg.email.sender, "robot@example.com")
            self.assertEqual(cfg.email.smtp_host, "smtp.example.com")
            self.assertEqual(cfg.email.smtp_port, 465)
            self.assertEqual(cfg.email.smtp_encryption, "tls")

    def test_cli_overrides_config_file(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "reachycheese.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "camera_source": "webcam",
                        "camera_index": 2,
                        "controller": "v1",
                        "email": {"to": "person@example.com"},
                    }
                ),
                encoding="utf-8",
            )

            cfg = parse_args(
                [
                    "--config",
                    str(cfg_path),
                    "--camera-source",
                    "reachy",
                    "--camera-index",
                    "0",
                    "--controller",
                    "v2",
                    "--email-to",
                    "override@example.com",
                ]
            )

            self.assertEqual(cfg.camera_source, "reachy")
            self.assertEqual(cfg.camera_index, 0)
            self.assertEqual(cfg.controller, "v2")
            self.assertEqual(cfg.email.recipient, "override@example.com")


if __name__ == "__main__":
    unittest.main()
