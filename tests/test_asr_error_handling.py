import unittest

from ReachyCheese import ReachyCheeseApp


class AsrErrorHandlingTests(unittest.TestCase):
    def test_should_disable_asr_on_fatal_portaudio_errors(self):
        exc = RuntimeError("Unanticipated host error [PaErrorCode -9999]: '[host API not founds error 0]'")
        self.assertTrue(ReachyCheeseApp._should_disable_asr_on_error(exc))

    def test_should_not_disable_asr_on_transient_error(self):
        exc = RuntimeError("temporary timeout")
        self.assertFalse(ReachyCheeseApp._should_disable_asr_on_error(exc))


if __name__ == "__main__":
    unittest.main()
