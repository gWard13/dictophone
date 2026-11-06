"""Conservative automatic reassignment of short turns using independent voice evidence."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .core import Turn


def refine_labels(turns: list[Turn], embeddings: list, max_duration_ms: int = 3000,
                  min_reference_ms: int = 3500, min_similarity: float = 0.6,
                  min_margin: float = 0.15) -> tuple[list[Turn], int]:
    """Leave-one-out centroids; never use the target itself to justify its label.

    Scores are cosine similarities, not calibrated probabilities. Never add a
    speaker, change time boundaries, use text, or force the number of clusters.
    All decisions use the original labels to avoid order-dependent propagation.
    """
    if len(turns) != len(embeddings):
        raise ValueError("One embedding slot is required per turn")
    if (type(max_duration_ms) is not int or max_duration_ms <= 0 or
            type(min_reference_ms) is not int or min_reference_ms <= 0 or
            not np.isfinite(min_similarity) or not -1 <= min_similarity <= 1 or
            not np.isfinite(min_margin) or not 0 <= min_margin <= 2):
        raise ValueError("Invalid refinement thresholds")
    vectors = []
    dimension = None
    for vector in embeddings:
        if vector is None:
            vectors.append(None)
            continue
        v = np.asarray(vector, dtype=np.float64)
        if v.ndim != 1 or not v.size or not np.isfinite(v).all() or np.linalg.norm(v) <= 0:
            raise ValueError("Invalid speaker embedding")
        if dimension is not None and v.size != dimension:
            raise ValueError("Inconsistent embedding dimensions")
        dimension = v.size
        vectors.append(v / np.linalg.norm(v))
    clean = [True] * len(turns)
    active = []
    for i in sorted(range(len(turns)), key=lambda j: turns[j].start_ms):
        active = [j for j in active if turns[j].end_ms > turns[i].start_ms]
        for j in active:
            if turns[j].speaker != turns[i].speaker:
                clean[i] = clean[j] = False
        active.append(i)
    total_sums, total_durations = {}, {}
    for j, reference in enumerate(turns):
        weight = reference.end_ms - reference.start_ms
        if not clean[j] or vectors[j] is None or weight <= 0:
            continue
        label = reference.speaker
        total_sums[label] = total_sums.get(label, np.zeros_like(vectors[j])) + vectors[j] * weight
        total_durations[label] = total_durations.get(label, 0) + weight
    result = list(turns)
    changed = 0
    for index, turn in enumerate(turns):
        if turn.end_ms - turn.start_ms > max_duration_ms or vectors[index] is None or not clean[index]:
            continue
        scores = []
        for label, total in total_sums.items():
            duration = total_durations[label]
            vector = total
            if label == turn.speaker:
                weight = turn.end_ms - turn.start_ms
                vector = total - vectors[index] * weight
                duration -= weight
            norm = np.linalg.norm(vector)
            if duration >= min_reference_ms and norm > 0:
                scores.append((float(vectors[index] @ (vector / norm)), label))
        scores.sort(key=lambda x: (-x[0], x[1]))
        if len(scores) < 2:
            continue
        best, second = scores[0], scores[1]
        if best[1] != turn.speaker and best[0] >= min_similarity and best[0] - second[0] >= min_margin:
            result[index] = replace(turn, speaker=best[1])
            changed += 1
    return result, changed
