# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
import numpy as np
import pytest

from observation_align import AlignmentError, StampedFrame, align_pending

CAMERAS = ("exterior_1", "exterior_2", "wrist")


def stamped(tag: int, t_us: int | None = None) -> StampedFrame:
    return StampedFrame(
        data=np.full((2, 2, 3), tag % 256, dtype=np.uint8),
        capture_time_us=t_us if t_us is not None else tag * 66_666,
    )


def test_join_by_shared_capture_time():
    pending = {
        "exterior_1": [stamped(10), stamped(11), stamped(12)],
        "exterior_2": [stamped(10), stamped(11), stamped(12)],
        "wrist": [stamped(10), stamped(11), stamped(12)],
    }
    cameras = align_pending(pending, cameras=CAMERAS)
    assert [int(f[0, 0, 0]) for f in cameras["wrist"]] == [10, 11, 12]
    assert all(not pending[name] for name in CAMERAS)


def test_lagging_wrist_waits_then_pairs():
    pending = {
        "exterior_1": [stamped(10), stamped(11), stamped(12)],
        "exterior_2": [stamped(10), stamped(11), stamped(12)],
        "wrist": [stamped(10)],  # lagging: only older stamp
    }
    cameras = align_pending(pending, cameras=CAMERAS)
    # Pairs at t=10; newer exterior frames stay pending for the next step.
    assert [int(f[0, 0, 0]) for f in cameras["exterior_1"]] == [10]
    assert [int(f[0, 0, 0]) for f in cameras["wrist"]] == [10]
    assert [int(f.data[0, 0, 0]) for f in pending["exterior_1"]] == [11, 12]
    assert pending["wrist"] == []


def test_unequal_pending_without_shared_stamp_refuses():
    pending = {
        "exterior_1": [stamped(10, 1_000_000)],
        "exterior_2": [stamped(11, 2_000_000)],
        "wrist": [stamped(12, 3_000_000)],
    }
    with pytest.raises(AlignmentError, match="no shared capture_time_us"):
        align_pending(pending, cameras=CAMERAS)


def test_unstamped_frame_refuses():
    pending = {
        "exterior_1": [stamped(1)],
        "exterior_2": [stamped(1)],
        "wrist": [StampedFrame(np.zeros((2, 2, 3), np.uint8), None)],
    }
    with pytest.raises(AlignmentError, match="capture_time_us"):
        align_pending(pending, cameras=CAMERAS)


def test_skew_within_one_ms_still_pairs():
    pending = {
        "exterior_1": [stamped(5, 1_000_000)],
        "exterior_2": [stamped(5, 1_000_500)],
        "wrist": [stamped(5, 999_500)],
    }
    cameras = align_pending(pending, cameras=CAMERAS, max_skew_us=1_000)
    assert all(len(cameras[name]) == 1 for name in CAMERAS)
