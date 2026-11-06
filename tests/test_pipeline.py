"""Fault-injection integration tests, not evidence of actual model accuracy."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dictophone.core import Turn, Word
from dictophone.pipeline import run, retry_llm


class Audio:
    class Mapping:
        def close(self):
            pass
    _mmap = Mapping()


class LLM:
    def __init__(self, *args):
        pass
    def punctuate(self, text):
        return text.capitalize() + "."
    def close(self):
        pass


class PipelineTests(unittest.TestCase):
    cfg = {"asr": {}, "diarization": {}, "llm": {}, "threads": 1}

    def setup_backends(self, diarization=None, asr=None, llm=LLM):
        self.enterContext(patch("dictophone.backends.decode_mp3", return_value=(Audio(), 2000)))
        self.enterContext(patch("dictophone.backends.transcribe", **(
            {"side_effect": asr} if isinstance(asr, Exception) else
            {"return_value": asr if asr is not None else [Word(100, 500, "привет")]})))
        self.enterContext(patch("dictophone.backends.diarize", **(
            {"side_effect": diarization} if isinstance(diarization, Exception) else
            {"return_value": diarization if diarization is not None else [Turn(0, 1000, "a")]})))
        self.enterContext(patch("dictophone.backends.LocalLLM", llm))

    def test_full_flow_and_retry_without_asr(self):
        self.setup_backends()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = run(root / "in.mp3", root / "out.json", self.cfg, session=root / "session.json")
            self.assertEqual(result["warnings"], [])
            row = json.loads((root / "out.json").read_text(encoding="utf-8"))["transcript"][0]
            self.assertEqual(row, {"speaker_id": "speaker_1", "start_ms": 100,
                                   "end_ms": 500, "text": "Привет."})
            with patch("dictophone.backends.transcribe", side_effect=AssertionError("Must not run")):
                retry_llm(root / "session.json", root / "retry.json", self.cfg)
            self.assertEqual((root / "out.json").read_bytes(), (root / "retry.json").read_bytes())

    def test_diarization_failure_is_unknown(self):
        self.setup_backends(diarization=RuntimeError("private diagnostic"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.json"
            result = run(Path(d) / "in.mp3", p, self.cfg)
            self.assertEqual(json.loads(p.read_text())["transcript"][0]["speaker_id"], "unknown")
            self.assertIn("diarization_failed:unknown_speakers", result["warnings"])

    def test_llm_failure_preserves_asr(self):
        def broken(*args):
            raise RuntimeError("private text")
        self.setup_backends(llm=broken)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.json"
            result = run(Path(d) / "in.mp3", p, self.cfg)
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["transcript"][0]["text"], "привет")
            self.assertEqual(result["llm_accepted_utterances"], 0)

    def test_asr_failure_does_not_export(self):
        self.setup_backends(asr=RuntimeError("ASR failed"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.json"
            with self.assertRaises(RuntimeError):
                run(Path(d) / "in.mp3", p, self.cfg)
            self.assertFalse(p.exists())

    def test_silence_exports_empty_array(self):
        self.setup_backends(asr=[], diarization=[])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.json"
            result = run(Path(d) / "in.mp3", p, self.cfg)
            self.assertEqual(json.loads(p.read_text()), {"transcript": []})
            self.assertFalse(result["speech_detected"])

    def test_unrecognized_detected_speech_has_marker(self):
        self.setup_backends(asr=[])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.json"
            run(Path(d) / "in.mp3", p, self.cfg)
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["transcript"][0]["text"], "[Неразборчиво]")

    def test_memory_error_is_fatal(self):
        self.setup_backends(diarization=MemoryError())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(MemoryError):
                run(Path(d) / "in.mp3", Path(d) / "out.json", self.cfg)
