"""Explicit ONLINE preparation step. Never invoked by the inference pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from urllib.parse import quote
from pathlib import Path


def download(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            total = 0
            next_progress = 32 * 1024 * 1024
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                total += len(chunk)
                if total >= next_progress:
                    print(f"  {destination.name}: {total // (1024 * 1024)} MiB", flush=True)
                    next_progress += 32 * 1024 * 1024
            expected = response.headers.get("Content-Length")
            if expected is not None and total != int(expected):
                raise IOError("Model download was truncated")
            if not total:
                raise IOError("Downloaded model is empty")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description="Download public model weights; no audio is read or sent")
    p.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    p.add_argument("--asr-revision", default="main", help="Use a commit hash for reproducible preparation")
    p.add_argument("--llm-revision", default="main")
    args = p.parse_args()
    # Plain HTTPS is easier to diagnose on restricted networks than Xet transport.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    asr = root / "faster-whisper-small"
    print("Downloading ASR", flush=True)
    asr_url = "https://huggingface.co/Systran/faster-whisper-small/resolve/" + quote(args.asr_revision, safe="") + "/"
    for name in ("config.json", "tokenizer.json", "vocabulary.txt", "README.md", "model.bin"):
        download(asr_url + name, asr / name)
    print("Downloading diarization", flush=True)
    release = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    archive = root / "segmentation.tar.bz2"
    download(release + "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2", archive)
    segmentation_dir = root / "segmentation"
    segmentation_dir.mkdir(exist_ok=True)
    # Extract just model.onnx, with no arbitrary archive paths or links.
    try:
        with tarfile.open(archive, "r:bz2") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and Path(m.name).name == "model.onnx"]
            if len(members) != 1:
                raise ValueError("Unexpected segmentation archive contents")
            with tar.extractfile(members[0]) as src, (segmentation_dir / "model.onnx").open("wb") as dst:
                shutil.copyfileobj(src, dst)
    finally:
        archive.unlink(missing_ok=True)
    download(release + "speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
             root / "embedding.onnx")
    print("Downloading local LLM", flush=True)
    llm_url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/" + quote(args.llm_revision, safe="") + "/"
    download(llm_url + "qwen2.5-0.5b-instruct-q4_k_m.gguf", root / "llm" / "model.gguf")
    files = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and ".cache" not in f.parts and f.name != "manifest.json":
            with f.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append({"path": str(f.relative_to(root)), "bytes": f.stat().st_size, "sha256": digest})
    manifest = {"asr_repository": "Systran/faster-whisper-small", "asr_revision": args.asr_revision,
                "llm_repository": "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "llm_revision": args.llm_revision,
                "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8", newline="\n")
    print("Models prepared. Check licenses before redistribution; inference can now run offline.")


if __name__ == "__main__":
    main()
