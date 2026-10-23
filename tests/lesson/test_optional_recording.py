"""Running the lesson with replay disabled must not pause or query the game."""

import unittest
from unittest.mock import patch

from bdxr_classroom.lesson.play import optional_recording


class OptionalRecording(unittest.TestCase):
    def test_default_leaves_live_game_untouched(self):
        with patch("bdxr_classroom.lesson.play.game_state") as state:
            self.assertIsNone(optional_recording(object(), 8765))
        state.assert_not_called()

    def test_enabled_replay_pauses_and_handles_missing_clip(self):
        with patch(
            "bdxr_classroom.lesson.play.pause_and_recordings", return_value=[]
        ) as pause, patch("builtins.print") as message:
            lab = object()
            self.assertIsNone(optional_recording(lab, 8765, enabled=True))
        pause.assert_called_once_with(lab, 8765)
        message.assert_called_once()
