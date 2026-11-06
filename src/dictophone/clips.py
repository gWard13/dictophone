"""Accurate local podcast clipping for reproducible evaluation."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from .core import atomic_json


def cut_mp3(source: Path, output: Path, start_ms: int, end_ms: int) -> dict:
    """Decode to mono 16 kHz, select samples, encode MP3 with a zero-based timeline.

    Re-encoding avoids inaccurate cuts at MP3 packet boundaries. No padding is
    invented when the requested end lies beyond the decoded source duration.
    """
    import av
    import numpy as np

    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("A clip must not overwrite the original recording")
    if source.suffix.lower() != ".mp3" or output.suffix.lower() != ".mp3":
        raise ValueError("Input and output must be MP3")
    if type(start_ms) is not int or type(end_ms) is not int or not 0 <= start_ms < end_ms:
        raise ValueError("Expected 0 <= start_ms < end_ms, in integer milliseconds")
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".clip-", suffix=".mp3", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    start_sample, end_sample = start_ms * 16, end_ms * 16
    cursor, written = 0, 0
    try:
        with av.open(str(source)) as src, av.open(str(temporary), "w", format="mp3") as dst:
            if not src.streams.audio:
                raise ValueError("No audio stream")
            if src.streams.audio[0].codec_context.name not in {"mp3", "mp3float", "mp3adu", "mp3adufloat"}:
                raise ValueError("Input is not MP3")
            stream = dst.add_stream("mp3", rate=16000)
            stream.layout = "mono"
            stream.bit_rate = 64000
            resampler = av.AudioResampler(format="s16p", layout="mono", rate=16000)

            def consume(frame):
                nonlocal cursor, written
                values = frame.to_ndarray()
                count = values.shape[1]
                a, b = max(start_sample, cursor), min(end_sample, cursor + count)
                if b > a:
                    data = np.ascontiguousarray(values[:, a - cursor:b - cursor])
                    clipped = av.AudioFrame.from_ndarray(data, format="s16p", layout="mono")
                    clipped.sample_rate = 16000
                    clipped.pts = written
                    for packet in stream.encode(clipped):
                        dst.mux(packet)
                    written += b - a
                cursor += count

            for frame in src.decode(audio=0):
                for converted in resampler.resample(frame):
                    consume(converted)
                if cursor >= end_sample:
                    break
            if cursor < end_sample:
                for converted in resampler.resample(None):
                    consume(converted)
            if written != end_sample - start_sample:
                raise ValueError("Requested clip extends beyond the recording")
            for packet in stream.encode(None):
                dst.mux(packet)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"start_ms": start_ms, "end_ms": end_ms, "duration_ms": end_ms - start_ms,
            "sample_rate": 16000, "channels": 1}


def prepare_clips(manifest: Path) -> Path:
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    source = (manifest.parent / data["source_file"]).resolve()
    directory = (manifest.parent / data["output_dir"]).resolve()
    clips = data["clips"]
    if not isinstance(clips, list) or not clips:
        raise ValueError("clips must be a non-empty list")
    seen = set()
    # Validate the full plan before producing any output.
    for clip in clips:
        name = clip["id"]
        a, b = clip["start_ms"], clip["end_ms"]
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name)
                or name in seen):
            raise ValueError("Invalid or duplicate clip ID")
        if type(a) is not int or type(b) is not int or not 0 <= a < b:
            raise ValueError("Invalid clip range")
        if (directory / f"{name}.mp3").resolve() == source:
            raise ValueError("Clip output would overwrite the original")
        seen.add(name)
    generated = []
    for clip in clips:
        output = directory / f"{clip['id']}.mp3"
        metadata = cut_mp3(source, output, clip["start_ms"], clip["end_ms"])
        generated.append({"id": clip["id"], "audio_file": output.name,
                          "source_offset_ms": metadata["start_ms"],
                          "duration_ms": metadata["duration_ms"],
                          "scenario": clip.get("scenario", "unspecified")})
    result = directory / "manifest.json"
    atomic_json(result, {"source_file": str(source), "source_url": data.get("source_url", ""),
                         "clips": generated})
    return result
