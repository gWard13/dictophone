"""Platform-independent alignment and output contract. Times are milliseconds."""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Word:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class Turn:
    start_ms: int
    end_ms: int
    speaker: str


@dataclass(frozen=True)
class Utterance:
    speaker_id: str
    start_ms: int
    end_ms: int
    text: str


def interval_ok(start: int, end: int, duration: int) -> bool:
    return type(start) is int and type(end) is int and 0 <= start <= end <= duration


def normalized_words(text: str) -> list[str]:
    # Keep digits and word boundaries: no additions, deletions, substitutions or reorderings.
    return re.findall(r"[^\W_]+(?:[-’'][^\W_]+)*", text.casefold(), re.UNICODE)


def guarded_text(original: str, candidate: object) -> str:
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("LLM returned an empty/non-string text")
    if normalized_words(original) != normalized_words(candidate):
        raise ValueError("LLM changed lexical content")
    if len(candidate) > max(64, len(original) * 3):
        raise ValueError("LLM returned excessive punctuation")
    # Avoid arbitrary symbols, control characters or markup in a supposedly punctuation-only edit.
    allowed = set(".,!?;:…—–-()[]«»\"'’ \t\n\r")
    if any(not (c.isalnum() or c.isspace() or c in allowed) for c in candidate):
        raise ValueError("LLM returned unsupported symbols")
    return " ".join(candidate.split())


def canonical_turns(turns: list[Turn], duration_ms: int) -> list[Turn]:
    valid = sorted(turns, key=lambda t: (t.start_ms, t.end_ms, t.speaker))
    mapping: dict[str, str] = {}
    result = []
    for t in valid:
        if not interval_ok(t.start_ms, t.end_ms, duration_ms) or not t.speaker:
            raise ValueError("Invalid diarization interval")
        if t.end_ms == t.start_ms:
            continue
        mapping.setdefault(t.speaker, f"speaker_{len(mapping) + 1}")
        result.append(Turn(t.start_ms, t.end_ms, mapping[t.speaker]))
    return result


def choose_speaker(word: Word, turns: list[Turn], min_overlap_ms: int = 40,
                   min_overlap_ratio: float = 0.2) -> str:
    """Assign maximum union coverage; concurrent distinct voices yield overlap.

    A boundary between consecutive speakers is not simultaneous speech. A word
    with no intersecting speech segment is unknown: never guess nearest voice.
    """
    start, end = word.start_ms, max(word.end_ms, word.start_ms + 1)
    events: dict[int, list[tuple[str, int]]] = {}
    for t in turns:
        a, b = max(start, t.start_ms), min(end, t.end_ms)
        if b > a:
            events.setdefault(a, []).append((t.speaker, 1))
            events.setdefault(b, []).append((t.speaker, -1))
    active: dict[str, int] = {}
    coverage: dict[str, int] = {}
    overlap = 0
    previous = start
    for at, changes in sorted(events.items()):
        width = at - previous
        speakers = [s for s, count in active.items() if count > 0]
        for s in speakers:
            coverage[s] = coverage.get(s, 0) + width
        if len(speakers) > 1:
            overlap += width
        for s, delta in changes:
            active[s] = active.get(s, 0) + delta
        previous = at
    if overlap >= min_overlap_ms and overlap / (end - start) >= min_overlap_ratio:
        return "overlap"
    if not coverage:
        return "unknown"
    return min(coverage, key=lambda s: (-coverage[s], s))


def align(words: list[Word], turns: list[Turn], duration_ms: int,
          pause_ms: int = 700, max_utterance_ms: int = 15000) -> list[Utterance]:
    turns = canonical_turns(turns, duration_ms)
    ordered = sorted(words, key=lambda w: (w.start_ms, w.end_ms))
    result: list[Utterance] = []
    cursor = 0
    active_turns: list[Turn] = []
    for w in ordered:
        if not interval_ok(w.start_ms, w.end_ms, duration_ms):
            raise ValueError("Invalid ASR timestamp")
        if not w.text.strip():
            continue
        while cursor < len(turns) and turns[cursor].start_ms < max(w.end_ms, w.start_ms + 1):
            active_turns.append(turns[cursor])
            cursor += 1
        active_turns = [t for t in active_turns if t.end_ms > w.start_ms]
        speaker = choose_speaker(w, active_turns)
        if (result and result[-1].speaker_id == speaker
                and w.start_ms - result[-1].end_ms <= pause_ms
                and w.end_ms - result[-1].start_ms <= max_utterance_ms):
            p = result[-1]
            result[-1] = Utterance(speaker, p.start_ms, max(p.end_ms, w.end_ms),
                                    p.text + " " + w.text.strip())
        else:
            result.append(Utterance(speaker, w.start_ms, w.end_ms, w.text.strip()))
    return result


def validate_transcript(data: object, duration_ms: int | None = None) -> None:
    if not isinstance(data, dict) or set(data) != {"transcript"}:
        raise ValueError("Expected only the transcript field")
    if not isinstance(data["transcript"], list):
        raise ValueError("transcript must be an array")
    previous = -1
    for row in data["transcript"]:
        if not isinstance(row, dict) or set(row) != {"speaker_id", "start_ms", "end_ms", "text"}:
            raise ValueError("Invalid transcript item fields")
        if not isinstance(row["speaker_id"], str) or not re.fullmatch(
                r"speaker_[1-9]\d*|unknown|overlap", row["speaker_id"]):
            raise ValueError("Invalid speaker_id")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("text must be non-empty")
        a, b = row["start_ms"], row["end_ms"]
        limit = duration_ms if duration_ms is not None else 2**63 - 1
        if not interval_ok(a, b, limit) or a < previous:
            raise ValueError("Invalid or unsorted timestamps")
        previous = a


def atomic_json(path: Path, data: object) -> None:
    """Write UTF-8/LF without BOM, preserving an existing export on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".dictophone-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export(path: Path, rows: list[Utterance], duration_ms: int) -> None:
    data = {"transcript": [asdict(r) for r in rows]}
    validate_transcript(data, duration_ms)
    atomic_json(path, data)


def wer(reference: str, hypothesis: str) -> dict:
    ref, hyp = normalized_words(reference), normalized_words(hypothesis)
    dp = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, dp[j] + 1, dp[j - 1] + (a != b)))
        dp = current
    return {"errors": dp[-1], "reference_words": len(ref),
            "wer": dp[-1] / len(ref) if ref else (0.0 if not hyp else None)}
