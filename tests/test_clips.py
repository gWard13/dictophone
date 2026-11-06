import tempfile
import unittest
from pathlib import Path

try:
    import av
    import numpy as np
except ImportError:
    av = None

from dictophone.backends import decode_mp3
from dictophone.clips import cut_mp3


@unittest.skipIf(av is None, "PyAV and NumPy are not installed")
class ClipTests(unittest.TestCase):
    def make_audio(self, path):
        with av.open(str(path), "w", format="mp3") as container:
            stream = container.add_stream("mp3", rate=16000)
            stream.layout = "mono"
            t = np.arange(32000) / 16000
            # The second second is silent, so offset accuracy can be checked by energy.
            signal = np.where(t < 1, np.sin(2 * np.pi * 440 * t) * 8000, 0)
            frame = av.AudioFrame.from_ndarray(signal.astype(np.int16)[None, :],
                                               format="s16p", layout="mono")
            frame.sample_rate = 16000
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)

    def test_cut_length_offset_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source.mp3"
            self.make_audio(source)
            before = source.read_bytes()
            output = root / "clip.mp3"
            cut_mp3(source, output, 1200, 1700)
            audio, duration = decode_mp3(output, root / "audio.f32")
            try:
                self.assertEqual(duration, 500)
                self.assertLess(float(np.max(np.abs(audio))), 0.01)
            finally:
                audio._mmap.close()
            self.assertEqual(source.read_bytes(), before)

    def test_short_source_does_not_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, output = root / "in.mp3", root / "out.mp3"
            self.make_audio(source)
            output.write_bytes(b"previous")
            with self.assertRaises(ValueError):
                cut_mp3(source, output, 1000, 3000)
            self.assertEqual(output.read_bytes(), b"previous")
            self.assertEqual(list(root.glob(".clip-*")), [])

    def test_original_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "in.mp3"
            self.make_audio(p)
            before = p.read_bytes()
            with self.assertRaises(ValueError):
                cut_mp3(p, p, 0, 500)
            self.assertEqual(p.read_bytes(), before)
