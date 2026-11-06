"""Real MP3 codec smoke test. Synthetic tones are not a speech accuracy test."""
import tempfile
import unittest
from pathlib import Path

try:
    import av
    import numpy as np
except ImportError:
    av = None

from dictophone.backends import decode_mp3


@unittest.skipIf(av is None, "PyAV and NumPy are not installed")
class DecodeTests(unittest.TestCase):
    def test_stereo_44100_mp3_to_mono_16000(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "tone.MP3"
            with av.open(str(source), "w", format="mp3") as container:
                stream = container.add_stream("mp3", rate=44100)
                stream.layout = "stereo"
                for i in range(10):
                    t = np.arange(4410) / 44100 + i / 10
                    signal = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)])
                    frame = av.AudioFrame.from_ndarray((signal * 8000).astype(np.int16),
                                                       format="s16p", layout="stereo")
                    frame.sample_rate = 44100
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
            before = source.read_bytes()
            audio, duration = decode_mp3(source, root / "audio.f32")
            try:
                self.assertEqual(audio.ndim, 1)
                self.assertEqual(audio.dtype, np.float32)
                self.assertTrue(np.isfinite(audio).all())
                self.assertLess(abs(duration - 1000), 100)
                self.assertAlmostEqual(len(audio) / 16000, duration / 1000, places=2)
            finally:
                audio._mmap.close()
            self.assertEqual(source.read_bytes(), before)

    def test_corrupt_mp3_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "broken.mp3"
            p.write_bytes(b"not a real mp3")
            with self.assertRaises(Exception):
                decode_mp3(p, Path(d) / "audio.f32")
