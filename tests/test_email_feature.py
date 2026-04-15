import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ReachyCheese import EmailConfig, PhotoEmailSender


class _FakeSMTP:
    def __init__(self, host, port, timeout=30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.calls = []

    def __enter__(self):
        self.calls.append("enter")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.calls.append("exit")
        return False

    def ehlo(self):
        self.calls.append("ehlo")

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, msg):
        self.calls.append(("send_message", msg["To"], msg["From"], msg["Subject"]))


class _FakeSMTP_SSL(_FakeSMTP):
    pass


class EmailFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.photo = Path(self.tmpdir.name) / "sample.png"
        self.photo.write_bytes(b"fake-image-bytes")

    def test_sender_disabled_when_incomplete_config(self):
        cfg = EmailConfig(recipient="", sender="", smtp_host="smtp.gmail.com", smtp_port=587)
        sender = PhotoEmailSender(cfg)
        self.assertFalse(sender.enabled)

    def test_send_photo_uses_starttls_and_succeeds(self):
        cfg = EmailConfig(
            recipient="to@example.com",
            sender="from@example.com",
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            smtp_encryption="start-tls",
            smtp_username="from@example.com",
            smtp_password="app-password",
            subject="Reachy photo",
        )
        sender = PhotoEmailSender(cfg)

        smtp_instance = _FakeSMTP("smtp.gmail.com", 587)
        with patch("ReachyCheese.smtplib.SMTP", return_value=smtp_instance):
            ok, err = sender.send_photo(self.photo)

        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertIn("starttls", smtp_instance.calls)
        self.assertIn(("login", "from@example.com", "app-password"), smtp_instance.calls)

    def test_send_photo_uses_tls_ssl_class(self):
        cfg = EmailConfig(
            recipient="to@example.com",
            sender="from@example.com",
            smtp_host="smtp.gmail.com",
            smtp_port=465,
            smtp_encryption="tls",
            smtp_username="from@example.com",
            smtp_password="app-password",
            subject="Reachy photo",
        )
        sender = PhotoEmailSender(cfg)

        smtp_ssl_instance = _FakeSMTP_SSL("smtp.gmail.com", 465)
        with patch("ReachyCheese.smtplib.SMTP_SSL", return_value=smtp_ssl_instance):
            ok, err = sender.send_photo(self.photo)

        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertNotIn("starttls", smtp_ssl_instance.calls)

    def test_send_photo_without_recipient_fails(self):
        cfg = EmailConfig(
            recipient="",
            sender="from@example.com",
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            smtp_encryption="start-tls",
            smtp_username="from@example.com",
            smtp_password="app-password",
        )
        sender = PhotoEmailSender(cfg)
        ok, err = sender.send_photo(self.photo)
        self.assertFalse(ok)
        self.assertIn("recipient", err)

    def test_send_photo_with_override_recipient_succeeds(self):
        cfg = EmailConfig(
            recipient="",
            sender="from@example.com",
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            smtp_encryption="start-tls",
            smtp_username="from@example.com",
            smtp_password="app-password",
            subject="Reachy photo",
        )
        sender = PhotoEmailSender(cfg)

        smtp_instance = _FakeSMTP("smtp.gmail.com", 587)
        with patch("ReachyCheese.smtplib.SMTP", return_value=smtp_instance):
            ok, err = sender.send_photo(self.photo, recipient_override="to@example.com")

        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertIn(("send_message", "to@example.com", "from@example.com", "Reachy photo"), smtp_instance.calls)

