from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLUSTER_THRESHOLD = 0.42
DEFAULT_DETECTION_SCORE_THRESHOLD = 0.80


@dataclass
class Face:
    box: tuple[float, float, float, float]
    person: int | None = None
    confident: bool = False
    feature: np.ndarray | None = None
    score: float = 1.0
    track_id: int | None = None


@dataclass
class Person:
    feature: np.ndarray
    thumbnail: np.ndarray
    first_frame: int
    count: int = 1


@dataclass
class Analysis:
    path: Path
    fingerprint: str
    width: int
    height: int
    fps: float
    frames: list[list[Face]] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    people: list[Person] = field(default_factory=list)
    hlg: bool = False


def identify(feature: np.ndarray, people: list[Person]) -> tuple[int | None, bool]:
    """A cluster match alone must never authorize revealing a face."""
    if not people:
        return None, False
    scores = np.array([float(feature @ person.feature) for person in people])
    best = int(scores.argmax())
    runner_up = float(np.partition(scores, -2)[-2]) if len(scores) > 1 else -1.0
    if scores[best] < CLUSTER_THRESHOLD or scores[best] - runner_up < 0.10:
        return None, False
    return best, bool(scores[best] >= 0.75)


def hidden_faces(
    faces: list[Face],
    revealed: set[int],
    *,
    score_threshold: float = DEFAULT_DETECTION_SCORE_THRESHOLD,
    excluded_tracks: set[int] | None = None,
) -> list[Face]:
    excluded_tracks = excluded_tracks or set()
    return [
        face
        for face in faces
        if face.score >= score_threshold
        and face.track_id not in excluded_tracks
        and not (face.confident and face.person in revealed)
    ]


def iou(a: tuple, b: tuple) -> float:
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - x) * max(0, bottom - y)
    return intersection / max(1, a[2] * a[3] + b[2] * b[3] - intersection)
