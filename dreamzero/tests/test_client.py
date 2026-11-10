# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
import json

import numpy as np
import pytest
from policy_client import action_chunk


def test_flat_enveloped_and_dual_metadata():
    flat = {
        "actions": np.ones((24, 8)).tolist(),
        "chunk_index": 0,
        "inference_seconds": 0.1,
    }
    for payload in (
        flat,
        {"type": "action_chunk", "data": flat},
        {"data": flat, **flat},
    ):
        assert action_chunk(json.dumps(payload).encode()) == flat


def test_rejects_conflicting_metadata():
    flat = {
        "actions": np.ones((24, 8)).tolist(),
        "chunk_index": 0,
        "inference_seconds": 0.1,
    }
    with pytest.raises(ValueError, match="conflicting"):
        action_chunk({"data": flat, **flat, "chunk_index": 1})


def test_rejects_nonfinite_actions():
    with pytest.raises(ValueError, match="nonfinite"):
        action_chunk({"actions": [[float("nan")] * 8], "chunk_index": 0})


def test_streamed_message_integral_index_is_normalized():
    result = action_chunk({"actions": [[0.0] * 8], "chunk_index": 2.0})
    assert result["chunk_index"] == 2
    assert isinstance(result["chunk_index"], int)


@pytest.mark.parametrize("index", [0.5, float("nan"), True, -1])
def test_invalid_indices_are_rejected(index):
    with pytest.raises(ValueError, match="invalid chunk index"):
        action_chunk({"actions": [[0.0] * 8], "chunk_index": index})
