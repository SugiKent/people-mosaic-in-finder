"""Reduce cloud jitter without changing face identity or leaving raw faces uncovered."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .domain import Face

POSITION_TAU = 0.25
SHRINK_TAU = 0.45
MAX_GAP = 0.25
SHRINK_HOLD = 0.6
SIZE_WINDOW = 0.6


@dataclass
class _Track:
    raw: tuple[float, float, float, float]
    centre: tuple[float, float]
    size: tuple[float, float]
    time: float
    person: int | None
    small_since: tuple[float | None, float | None] = (None, None)
    size_history: tuple[tuple[float, float, float], ...] = ()
    track_id: int | None = None
    confident: bool = False


def _continue_exclusion(face, previous, faces, tracks, timestamp):
    """Be stricter than smoothing when carrying a user's object exclusion."""
    if (
        previous is None
        or face.person != previous.person
        or (face.confident and not previous.confident)
    ):
        return False
    cx, cy = _centre(face.box)
    px, py = _centre(previous.raw)
    # A nearby second detection/track makes an object's correspondence ambiguous.
    # Keep smoothing but start new exclusion IDs rather than transferring permission.
    for other in faces:
        if other is not face and _valid(other.box):
            ox, oy = _centre(other.box)
            if math.hypot(ox - cx, oy - cy) < 1.6 * max(*face.box[2:], *other.box[2:]):
                return False
    for other in tracks:
        if other is not previous and 0 < timestamp - other.time <= MAX_GAP:
            ox, oy = _centre(other.raw)
            if math.hypot(ox - px, oy - py) < 1.6 * max(*previous.raw[2:], *other.raw[2:]):
                return False
    return True


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
        width, height = (dimension * 1.15 for dimension in raw_size)
        box = (raw_centre[0] - width / 2, raw_centre[1] - height / 2, width, height)
        return replace(face, box=box), _Track(
            face.box,
            raw_centre,
            (width, height),
            timestamp,
            face.person,
            size_history=((timestamp, *raw_size),),
        )
    dt = timestamp - track.time
    deadband = max(6.0, min(24.0, min(raw_size) * 0.15))
    distance = math.dist(raw_centre, track.centre)
    # Adapt continuously to large movement. Discrete "smooth/fast/snap"
    # thresholds can alternate across adjacent frames and add visible jitter.
    relative_motion = distance / min(raw_size)
    alpha = -math.expm1(-dt / POSITION_TAU * (1 + 20 * relative_motion**2))
    centre = tuple(
        previous if abs(current - previous) <= deadband else previous + alpha * (current - previous)
        for previous, current in zip(track.centre, raw_centre, strict=True)
    )
    history = tuple(record for record in track.size_history if timestamp - record[0] <= SIZE_WINDOW)
    history += ((timestamp, *raw_size),)
    recent_max = (max(record[1] for record in history), max(record[2] for record in history))
    # Bound lag before sizing: a wider static deadband must not make a cloud
    # grow without limit or fail to cover a small fast-moving face.
    centre = tuple(
        current - max(-dimension * 0.25, min(dimension * 0.25, current - stable))
        for current, stable, dimension in zip(raw_centre, centre, recent_max, strict=True)
    )
    shrink = -math.expm1(-dt / SHRINK_TAU)
    size, small_since = [], []
    for dimension, (previous, current, raw_position, stable_position) in enumerate(
        zip(track.size, raw_size, raw_centre, centre, strict=True)
    ):
        # A box centred at the filtered position must contain the whole raw
        # box. Its opaque PNG core therefore continues to cover the real face.
        required = max(current * 1.15, current + 2 * abs(raw_position - stable_position))
        since = None
        if required >= previous:
            value = required
        elif previous - required <= max(5.0, previous * 0.12):
            value = previous
        else:
            since = track.small_since[dimension]
            since = timestamp if since is None else since
            value = previous
            if timestamp - since >= SHRINK_HOLD:
                value = max(required, previous + shrink * (required - previous))
        # A rolling upper bound cannot echo each raw size fluctuation back into
        # the cloud. A genuinely smaller face still releases the old bound in
        # 0.6 seconds, preventing indefinitely oversized masks.
        size.append(min(value, max(required, recent_max[dimension] * 1.5)))
        small_since.append(since)
    width, height = size
    box = (centre[0] - width / 2, centre[1] - height / 2, width, height)
    stable = replace(face, box=box)
    return stable, _Track(
        face.box,
        centre,
        (width, height),
        timestamp,
        face.person,
        tuple(small_since),
        history,
    )


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
    next_id = 0
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
            previous = matched.get(current)
            if _continue_exclusion(face, previous, faces, tracks, timestamp):
                track.track_id = previous.track_id
            else:
                track.track_id = next_id
                next_id += 1
            stable.track_id = track.track_id
            track.confident = face.confident
            output.append(stable)
            next_tracks.append(track)
        result.append(output)
        tracks, previous_time = next_tracks, timestamp
    return result
