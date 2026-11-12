# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Align multi-camera pending frames by capture_time_us.

Independent per-track InputBuffers deliver frames out of lockstep. Pairing by
arrival index misaligns exterior/wrist windows with robot state. This module
joins stamped frames on a shared capture time (microseconds) within a small
skew, and leaves unmatched newer pendings so a lagging track can catch up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 1 ms: same capture moment across independently delivered tracks.
DEFAULT_SKEW_US = 1_000


@dataclass(frozen=True, eq=False)
class StampedFrame:
    """One camera frame with the sender's capture time, if any.

    Equality is identity-based: ``data`` is an ndarray and must not be compared
    elementwise when locating a frame in a pending list.
    """

    data: np.ndarray
    capture_time_us: int | None

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


class AlignmentError(ValueError):
    """Frames cannot be paired across cameras."""


def align_pending(
    pending: dict[str, list[StampedFrame]],
    *,
    cameras: tuple[str, ...],
    max_skew_us: int = DEFAULT_SKEW_US,
    max_keep: int = 4,
) -> dict[str, tuple[np.ndarray, ...]]:
    """Return one co-timed window per camera, mutating *pending* in place.

    Picks the newest capture time that every camera can match within
    *max_skew_us*, taking up to *max_keep* frames ending at that moment from
    each track. Frames newer than the matched window stay in *pending*.
    Older unmatched frames are dropped.

    Raises:
        AlignmentError: A camera has no frames, a frame is unstamped, or no
            shared capture time exists within the skew.
    """
    for name in cameras:
        frames = pending.get(name) or []
        if not frames:
            raise AlignmentError(f"waiting for camera frames: {name}")
        if any(f.capture_time_us is None for f in frames):
            raise AlignmentError(
                f"camera {name} frames lack capture_time_us; cannot align"
            )

    # Candidate anchors: newest stamp on the first camera, then older.
    primary = cameras[0]
    anchors = sorted(
        {int(f.capture_time_us) for f in pending[primary]},  # type: ignore[arg-type]
        reverse=True,
    )
    for anchor in anchors:
        matched: dict[str, list[StampedFrame]] = {}
        ok = True
        for name in cameras:
            hit = _best_match(pending[name], anchor, max_skew_us)
            if hit is None:
                ok = False
                break
            # Window ending at the matched frame, oldest first, capped.
            end = pending[name].index(hit)
            start = max(0, end + 1 - max_keep)
            matched[name] = pending[name][start : end + 1]
        if not ok:
            continue
        # Drop through the matched frame; keep strictly newer pendings.
        for name in cameras:
            cut = pending[name].index(matched[name][-1]) + 1
            pending[name] = pending[name][cut:]
        return {name: tuple(f.data for f in matched[name]) for name in cameras}

    raise AlignmentError(
        "no shared capture_time_us across cameras within "
        f"{max_skew_us} us skew"
    )


def _best_match(
    frames: list[StampedFrame], anchor_us: int, max_skew_us: int
) -> StampedFrame | None:
    """Closest frame to *anchor_us* within skew; prefer the newest on a tie."""
    best: StampedFrame | None = None
    best_skew = max_skew_us + 1
    best_i = -1
    for i, frame in enumerate(frames):
        assert frame.capture_time_us is not None
        skew = abs(int(frame.capture_time_us) - anchor_us)
        if skew > max_skew_us:
            continue
        if skew < best_skew or (skew == best_skew and i > best_i):
            best = frame
            best_skew = skew
            best_i = i
    return best
