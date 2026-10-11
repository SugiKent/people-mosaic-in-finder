"""Reduce cloud jitter without changing face identity or leaving raw faces uncovered."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .domain import Face

POSITION_TAU = 0.14
SHRINK_TAU = 0.28
MAX_GAP = 0.25


@dataclass
class _Track:
    raw: tuple[float, float, float, float]
    centre: tuple[float, float]
    size: tuple[float, float]
    time: float
    person: int | None


def _centre(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


def _valid(box):
    return all(math.isfinite(value) for value in box) and box[2] > 0 and box[3] > 0


def _matches(faces, tracks, timestamp):
    """Greedy nearest geometry, with each old track used at most once."""
    candidates = []
    for current, face in enumerate(faces):
        if not _valid(face.box):
            continue
        cx, cy = _centre(face.box)
        w, h = face.box[2:]
        for previous, track in enumerate(tracks):
            gap = timestamp - track.time
            if not 0 < gap <= MAX_GAP:
                continue
            px, py = _centre(track.raw)
            pw, ph = track.raw[2:]
            if not (0.5 <= w / pw <= 2 and 0.5 <= h / ph <= 2):
                continue
            distance = math.hypot(cx - px, cy - py)
            size = max(w, h, pw, ph)
            if distance > size * 0.8:
                continue
            # Identity only breaks close geometric ties. It is never copied to
            # a current Face and never creates permission to show a face.
            affinity = 0.03 if face.person is not None and face.person == track.person else 0
            candidates.append((distance / size - affinity, current, previous))
    used_current, used_previous, matched = set(), set(), {}
    for _, current, previous in sorted(candidates):
        if current not in used_current and previous not in used_previous:
            matched[current] = tracks[previous]
            used_current.add(current)
            used_previous.add(previous)
    return matched


def _advance(face, timestamp, track):
    raw_centre = _centre(face.box)
    raw_size = face.box[2:]
    if track is None:
        width, height = (dimension * 1.08 for dimension in raw_size)
        box = (raw_centre[0] - width / 2, raw_centre[1] - height / 2, width, height)
        return replace(face, box=box), _Track(
            face.box, raw_centre, (width, height), timestamp, face.person
        )
    dt = timestamp - track.time
    deadband = max(3.0, min(12.0, min(raw_size) * 0.08))
    distance = math.dist(raw_centre, track.centre)
    alpha = -math.expm1(-dt / POSITION_TAU)
    # Follow deliberate movement rapidly so smoothing does not create a large
    # trailing cloud. Tiny changes hold still; moderate changes use time-based EMA.
    if distance > min(raw_size) * 0.35:
        alpha = 1.0
    elif distance > min(raw_size) * 0.24:
        alpha = max(alpha, 0.65)
    centre = tuple(
        previous if abs(current - previous) <= deadband else previous + alpha * (current - previous)
        for previous, current in zip(track.centre, raw_centre, strict=True)
    )
    # Bound lag before sizing: a wider static deadband must not make a cloud
    # grow without limit or fail to cover a small fast-moving face.
    centre = tuple(
        current - max(-dimension * 0.15, min(dimension * 0.15, current - stable))
        for current, stable, dimension in zip(raw_centre, centre, raw_size, strict=True)
    )
    shrink = -math.expm1(-dt / SHRINK_TAU)
    size = []
    for previous, current, raw_position, stable_position in zip(
        track.size, raw_size, raw_centre, centre, strict=True
    ):
        # A box centred at the filtered position must contain the whole raw
        # box. Its opaque PNG core therefore continues to cover the real face.
        required = max(current * 1.08, current + 2 * abs(raw_position - stable_position))
        if required >= previous:
            value = required
        elif previous - required <= max(3.0, previous * 0.08):
            value = previous
        else:
            value = max(required, previous + shrink * (required - previous))
        # A stale large detection must not keep an oversized cloud after a
        # face becomes smaller. Preserve the required coverage at every size.
        size.append(min(value, max(required, current * 1.3)))
    width, height = size
    box = (centre[0] - width / 2, centre[1] - height / 2, width, height)
    stable = replace(face, box=box)
    return stable, _Track(face.box, centre, (width, height), timestamp, face.person)


def stabilize_faces(
    frames: Sequence[list[Face]],
    times: Sequence[float],
    cuts: set[int] | None = None,
) -> list[list[Face]]:
    """Return matching frame/order/identity metadata with steadier masking boxes.

    No new faces are emitted and unmatched tracks disappear immediately. Empty
    frames, scene cuts, nonmonotonic timestamps, or a long gap reset association.
    Manual masking regions are intentionally outside this post-analysis step.
    """
    tracks, result = [], []
    cuts = cuts or set()
    previous_time = None
    for index, (faces, timestamp) in enumerate(zip(frames, times, strict=True)):
        if index in cuts or previous_time is None or not 0 < timestamp - previous_time <= MAX_GAP:
            tracks = []
        matched = _matches(faces, tracks, timestamp)
        output, next_tracks = [], []
        for current, face in enumerate(faces):
            if not _valid(face.box):
                output.append(face)
                continue
            stable, track = _advance(face, timestamp, matched.get(current))
            output.append(stable)
            next_tracks.append(track)
        result.append(output)
        tracks, previous_time = next_tracks, timestamp
    return result
