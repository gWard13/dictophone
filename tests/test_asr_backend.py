import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import numpy as np

from dictophone.backends import transcribe
from dictophone.core import Turn


class TimestampOffsetTests(unittest.TestCase):
    def test_cropped_asr_restores_global_timeline(self):
        class FakeWhisper:
            def __init__(self, *args, **kwargs):
                pass
            def transcribe(self, *args, **kwargs):
                word=SimpleNamespace(start=0.1,end=0.3,word='да')
                return iter([SimpleNamespace(text='да',words=[word])]),None
        module=ModuleType('faster_whisper')
        module.WhisperModel=FakeWhisper
        with tempfile.TemporaryDirectory() as d, patch.dict('sys.modules',{'faster_whisper':module}):
            for name in ['model.bin','config.json','tokenizer.json']:
                (Path(d)/name).write_bytes(b'model placeholder for adapter test')
            words=transcribe(np.zeros(96000,dtype=np.float32),6000,
                             {'model_dir':d,'speech_guided':True},'ru',1,
                             [Turn(1000,2000,'a'),Turn(4000,5000,'b')])
        self.assertEqual([(w.start_ms,w.end_ms) for w in words],[(600,800),(3600,3800)])
