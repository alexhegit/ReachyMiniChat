import unittest
from unittest.mock import patch

from ReachyCheese import PreviewGUI


class PreviewGUIShutdownTests(unittest.TestCase):
    def _mk_gui(self):
        gui = PreviewGUI.__new__(PreviewGUI)
        gui._ready = True
        gui._window_name = "ReachyCheese"
        return gui

    def test_is_running_returns_false_when_window_query_raises(self):
        gui = self._mk_gui()

        with patch("ReachyCheese.cv2.getWindowProperty", side_effect=Exception("window closed")):
            running = PreviewGUI.is_running(gui)

        self.assertFalse(running)
        self.assertFalse(gui._ready)

    def test_is_running_returns_false_when_window_not_visible(self):
        gui = self._mk_gui()

        with patch("ReachyCheese.cv2.getWindowProperty", return_value=0.0):
            running = PreviewGUI.is_running(gui)

        self.assertFalse(running)
        self.assertFalse(gui._ready)


if __name__ == "__main__":
    unittest.main()
