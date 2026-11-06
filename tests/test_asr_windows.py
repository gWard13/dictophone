import unittest
from dictophone.core import Turn, asr_windows


class SpeechWindowsTests(unittest.TestCase):
    def test_same_speaker_short_pauses_are_kept_as_context(self):
        self.assertEqual(asr_windows([Turn(1000,2000,'a'),Turn(3000,4000,'a')],6000),[(500,4500)])

    def test_change_of_speaker_has_disjoint_padding(self):
        windows=asr_windows([Turn(1000,2000,'a'),Turn(2200,3000,'b')],5000)
        self.assertEqual(windows,[(500,2100),(2100,3500)])

    def test_overlapping_speech_is_decoded_once(self):
        self.assertEqual(asr_windows([Turn(1000,3000,'a'),Turn(2000,4000,'b')],5000),[(500,4500)])

    def test_diarization_failure_falls_back_to_full_audio(self):
        self.assertEqual(asr_windows([],5000),[(0,5000)])

    def test_original_time_origin_and_bounds(self):
        self.assertEqual(asr_windows([Turn(0,1000,'a'),Turn(9000,10000,'a')],10000),[(0,1500),(8500,10000)])
