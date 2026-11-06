import json
import tempfile
import unittest
from pathlib import Path

from dictophone.core import (Turn, Utterance, Word, align, choose_speaker,
                              export, guarded_text, validate_transcript, wer)


class AlignmentTests(unittest.TestCase):
    def test_speakers_in_order_and_long_pause(self):
        turns = [Turn(0, 1000, "9"), Turn(1000, 2000, "3"), Turn(3000, 4000, "9")]
        words = [Word(100, 400, "Привет"), Word(500, 800, "друг"),
                 Word(1100, 1400, "Здравствуйте"), Word(3100, 3500, "Да")]
        result = align(words, turns, 4000)
        self.assertEqual([r.speaker_id for r in result], ["speaker_1", "speaker_2", "speaker_1"])
        self.assertEqual(result[0].text, "Привет друг")
        self.assertEqual(result[-1].start_ms, 3100)

    def test_sequential_boundary_is_not_overlap(self):
        turns = [Turn(0, 500, "speaker_1"), Turn(500, 1000, "speaker_2")]
        self.assertEqual(choose_speaker(Word(400, 700, "да"), turns), "speaker_2")

    def test_simultaneous_speech_is_overlap(self):
        turns = [Turn(0, 1000, "speaker_1"), Turn(300, 800, "speaker_2")]
        self.assertEqual(choose_speaker(Word(400, 700, "да"), turns), "overlap")

    def test_duplicate_same_voice_is_not_overlap(self):
        turns = [Turn(0, 1000, "speaker_1"), Turn(300, 800, "speaker_1")]
        self.assertEqual(choose_speaker(Word(400, 700, "да"), turns), "speaker_1")

    def test_uncovered_word_is_unknown(self):
        self.assertEqual(choose_speaker(Word(1000, 1200, "да"), [Turn(0, 500, "speaker_1")]), "unknown")

    def test_bad_word_time_rejected(self):
        with self.assertRaises(ValueError):
            align([Word(200, 100, "да")], [], 1000)

    def test_empty_recording(self):
        self.assertEqual(align([], [], 5000), [])

    def test_long_same_speaker_is_split(self):
        rows = align([Word(0, 500, "а"), Word(600, 1200, "б")], [Turn(0, 2000, "x")],
                     2000, max_utterance_ms=1000)
        self.assertEqual(len(rows), 2)


class LLMGuardTests(unittest.TestCase):
    def test_punctuation_and_case(self):
        self.assertEqual(guarded_text("привет как дела", "Привет! Как дела?"), "Привет! Как дела?")

    def test_hallucination_rejected(self):
        for candidate in ("Привет, мой друг", "привет", "дела как привет", "Привет как дела 42"):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                guarded_text("привет как дела", candidate)

    def test_symbol_injection_rejected(self):
        with self.assertRaises(ValueError):
            guarded_text("привет", "Привет <>")


class ExportTests(unittest.TestCase):
    def test_export_encoding_and_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "диалог.json"
            export(p, [Utterance("speaker_1", 0, 1000, 'Он сказал: "Да".\nДалее.')], 1000)
            data = p.read_bytes()
            self.assertFalse(data.startswith(b"\xef\xbb\xbf"))
            self.assertNotIn(b"\r\n", data)
            validate_transcript(json.loads(data), 1000)

    def test_invalid_output_does_not_replace_existing(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.json"
            p.write_text("original")
            with self.assertRaises(ValueError):
                export(p, [Utterance("speaker_1", 0, 1001, "да")], 1000)
            self.assertEqual(p.read_text(), "original")

    def test_boolean_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            validate_transcript({"transcript": [{"speaker_id": "unknown", "start_ms": False,
                                                "end_ms": 1, "text": "да"}]})

    def test_wer(self):
        self.assertEqual(wer("Добрый день!", "добрый вечер")["wer"], 0.5)
        self.assertIsNone(wer("", "вставка")["wer"])


if __name__ == "__main__":
    unittest.main()
