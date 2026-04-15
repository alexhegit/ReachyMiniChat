import unittest
from unittest.mock import patch

from ReachyCheese import EmailConfig, ReachyCheeseApp


class EmailPromptFeatureTests(unittest.TestCase):
    def test_parse_email_input_skip_and_empty(self):
        self.assertIsNone(ReachyCheeseApp.parse_email_recipient_input(""))
        self.assertIsNone(ReachyCheeseApp.parse_email_recipient_input("skip"))
        self.assertIsNone(ReachyCheeseApp.parse_email_recipient_input(" SKIP "))

    def test_parse_email_input_valid_address(self):
        self.assertEqual(
            ReachyCheeseApp.parse_email_recipient_input("heye_dev@163.com"),
            "heye_dev@163.com",
        )

    def test_parse_email_input_invalid_address(self):
        self.assertIsNone(ReachyCheeseApp.parse_email_recipient_input("not-an-email"))
        self.assertIsNone(ReachyCheeseApp.parse_email_recipient_input("abc@local"))

    def test_prompt_uses_default_on_empty_input(self):
        app = ReachyCheeseApp.__new__(ReachyCheeseApp)
        app.cfg = type("Cfg", (), {"email": EmailConfig(recipient="default@example.com")})()
        app._email_sender = type("Sender", (), {"enabled": True})()

        with patch("tkinter.Tk", side_effect=Exception("no-gui")):
            with patch("builtins.input", return_value=""):
                got = ReachyCheeseApp._prompt_email_recipient(app)
        self.assertEqual(got, "default@example.com")

    def test_prompt_skip_returns_none(self):
        app = ReachyCheeseApp.__new__(ReachyCheeseApp)
        app.cfg = type("Cfg", (), {"email": EmailConfig(recipient="default@example.com")})()
        app._email_sender = type("Sender", (), {"enabled": True})()

        with patch("tkinter.Tk", side_effect=Exception("no-gui")):
            with patch("builtins.input", return_value="skip"):
                got = ReachyCheeseApp._prompt_email_recipient(app)
        self.assertIsNone(got)

    def test_run_email_prompt_once_sets_pending_recipient(self):
        app = ReachyCheeseApp.__new__(ReachyCheeseApp)
        app._pending_email_recipient = None

        with patch.object(ReachyCheeseApp, "_prompt_email_recipient", return_value="person@example.com"):
            ReachyCheeseApp._run_email_prompt_once(app)

        self.assertEqual(app._pending_email_recipient, "person@example.com")


if __name__ == "__main__":
    unittest.main()
