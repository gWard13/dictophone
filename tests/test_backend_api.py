"""Constructor compatibility checks with the installed native libraries."""
import unittest

try:
    import sherpa_onnx as so
except ImportError:
    so = None


@unittest.skipIf(so is None, "sherpa-onnx is not installed")
class SherpaApiTests(unittest.TestCase):
    def test_diarization_config_constructor(self):
        config = so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model="missing.onnx", window_shift_ratio=0.1),
                num_threads=1, provider="cpu"),
            embedding=so.SpeakerEmbeddingExtractorConfig(
                model="missing.onnx", num_threads=1, provider="cpu"),
            clustering=so.FastClusteringConfig(num_clusters=-1, threshold=0.5),
            min_duration_on=0.2, min_duration_off=0.3)
        self.assertEqual(config.clustering.num_clusters, -1)
