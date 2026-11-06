from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

from . import backends
from .core import (Turn, Utterance, Word, align, atomic_json, canonical_turns,
                   export, validate_transcript)


def load_config(path: Path) -> dict:
    path = path.resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    # Relative model paths are relative to the config, not to the shell directory.
    for section, keys in {"asr": ["model_dir"], "diarization": ["segmentation_model", "embedding_model"],
                          "llm": ["model_path"]}.items():
        if section == "diarization" and "refinement_embedding_model" in cfg[section]:
            keys = keys + ["refinement_embedding_model"]
        for key in keys:
            value = cfg[section][key]
            if not isinstance(value, str) or not value:
                raise ValueError(f"Missing {section}.{key}")
            cfg[section][key] = str((path.parent / value).resolve())
    cfg["threads"] = int(cfg.get("threads", min(4, os.cpu_count() or 1)))
    if cfg["threads"] < 1:
        raise ValueError("threads must be positive")
    for key in ("pause_ms", "max_utterance_ms"):
        value = cfg.get("alignment", {}).get(key, 700 if key == "pause_ms" else 15000)
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid alignment.{key}")
    return cfg


def postprocess(raw: list[Utterance], cfg: dict, warnings: list[str]) -> tuple[list[Utterance], int]:
    if not raw:
        return [], 0
    model = None
    accepted = 0
    result = []
    try:
        model = backends.LocalLLM(cfg["llm"], cfg["threads"])
        for index, row in enumerate(raw):
            if row.text == "[Неразборчиво]":
                result.append(row)
                continue
            try:
                result.append(replace(row, text=model.punctuate(row.text)))
                accepted += 1
            except (MemoryError, OSError):
                raise
            except Exception:
                # Never include transcript text or raw native exception messages in diagnostics.
                warnings.append(f"llm_rejected_or_failed:utterance_{index + 1}")
                result.append(row)
        return result, accepted
    except (MemoryError, OSError) as exc:
        # Missing model is a recoverable LLM setup error; resource exhaustion is fatal.
        if isinstance(exc, MemoryError) or (isinstance(exc, OSError) and exc.errno in {12, 28}):
            raise
        warnings.append("llm_unavailable:original_asr_preserved")
        return raw, 0
    except Exception:
        warnings.append("llm_unavailable:original_asr_preserved")
        return raw, 0
    finally:
        if model is not None:
            model.close()


def add_unintelligible(words: list[Word], turns: list[Turn], duration_ms: int) -> list[Word]:
    """Add only whole diarized speech regions with no ASR words, not every word gap.

    A model cannot reliably tell silence from unintelligible speech in all cases;
    use positive diarization evidence and avoid fabricating any lexical content.
    """
    spans = sorted((t.start_ms, t.end_ms) for t in turns if t.end_ms > t.start_ms)
    regions: list[list[int]] = []
    for a, b in spans:
        if regions and a <= regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], b)
        else:
            regions.append([a, b])
    missing = []
    for a, b in regions:
        if b - a >= 200 and not any(w.start_ms < b and w.end_ms > a for w in words):
            missing.append(Word(a, b, "[Неразборчиво]"))
    return words + missing


def run(source: Path, output: Path, cfg: dict, language: str = "ru",
        session: Path | None = None, report: Path | None = None,
        progress=lambda stage: None) -> dict:
    started = time.perf_counter()
    warnings: list[str] = []
    timings: dict[str, float] = {}
    short_turn_changes = 0
    if source.resolve() == output.resolve():
        raise ValueError("Output must not overwrite source audio")
    # A pre-existing export is kept until a valid replacement is fully written.
    with tempfile.TemporaryDirectory(prefix="dictophone-") as temporary:
        progress("decode")
        t = time.perf_counter()
        audio, duration = backends.decode_mp3(source, Path(temporary) / "audio.f32")
        timings["decode"] = time.perf_counter() - t
        try:
            progress("diarization")
            t = time.perf_counter()
            try:
                turns = backends.diarize(audio, duration, cfg["diarization"], cfg["threads"])
                canonical_turns(turns, duration)  # validate within recoverable stage
                if cfg["diarization"].get("refine_short_turns", False):
                    try:
                        turns, short_turn_changes = backends.refine_short_turns(audio, turns, cfg["diarization"], cfg["threads"])
                    except MemoryError:
                        raise
                    except OSError as exc:
                        if exc.errno in {12, 28}:
                            raise
                        warnings.append("short_turn_refinement_failed:original_diarization_preserved")
                    except Exception:
                        warnings.append("short_turn_refinement_failed:original_diarization_preserved")
            except (MemoryError, OSError) as exc:
                if isinstance(exc, MemoryError) or (isinstance(exc, OSError) and exc.errno in {12, 28}):
                    raise
                turns = []
                warnings.append("diarization_failed:unknown_speakers")
            except Exception:
                turns = []
                warnings.append("diarization_failed:unknown_speakers")
            timings["diarization"] = time.perf_counter() - t
            gc.collect()
            progress("asr")
            t = time.perf_counter()
            words = backends.transcribe(audio, duration, cfg["asr"], language, cfg["threads"], turns=turns)
            timings["asr"] = time.perf_counter() - t
            if words and not turns and not any(w.startswith("diarization_failed") for w in warnings):
                warnings.append("diarization_empty:unknown_speakers")
        finally:
            # Explicit close required on Windows before TemporaryDirectory cleanup.
            audio._mmap.close()
            del audio
            gc.collect()
        progress("alignment")
        raw = align(add_unintelligible(words, turns, duration), turns, duration,
                    **cfg.get("alignment", {}))
        progress("llm")
        t = time.perf_counter()
        rows, accepted = postprocess(raw, cfg, warnings)
        timings["llm"] = time.perf_counter() - t
        if session:
            # Explicit opt-in: contains private text; not a diagnostic log.
            atomic_json(session, {"version": 1, "duration_ms": duration, "language": language,
                                  "raw_transcript": [asdict(r) for r in raw],
                                  "words": [asdict(w) for w in words],
                                  "turns": [asdict(t) for t in turns]})
        progress("export")
        export(output, rows, duration)
    total = time.perf_counter() - started
    metadata = {"duration_ms": duration, "processing_seconds": total,
                "rtf": total / (duration / 1000), "stage_seconds": timings,
                "utterances": len(rows), "speech_detected": bool(raw),
                "llm_accepted_utterances": accepted, "warnings": warnings}
    metadata["short_turn_reassignments"] = short_turn_changes
    if report:
        atomic_json(report, metadata)
    progress("complete")
    return metadata


def retry_llm(session: Path, output: Path, cfg: dict) -> dict:
    data = json.loads(session.read_text(encoding="utf-8-sig"))
    if data.get("version") != 1 or type(data.get("duration_ms")) is not int or data["duration_ms"] < 0:
        raise ValueError("Unsupported or invalid session")
    validate_transcript({"transcript": data["raw_transcript"]}, data["duration_ms"])
    raw = [Utterance(**row) for row in data["raw_transcript"]]
    warnings: list[str] = []
    rows, accepted = postprocess(raw, cfg, warnings)
    export(output, rows, data["duration_ms"])
    return {"llm_accepted_utterances": accepted, "warnings": warnings}
