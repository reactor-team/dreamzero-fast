# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dreamzero_model import DreamZeroInput, DreamZeroModel, InvalidRequest

CAMERAS = ("exterior_1", "exterior_2", "wrist")


def request(values=(1,), prompt="pick up the ball"):
    return DreamZeroInput(
        prompt,
        {
            name: tuple(np.full((2, 3, 3), value, dtype=np.uint8) for value in values)
            for name in CAMERAS
        },
        (0.1,) * 7,
        0.5,
    )


@pytest.fixture
def model():
    model = DreamZeroModel()
    model.rank = 0
    model._image_height, model._image_width = 2, 3
    model._action_head = SimpleNamespace(current_start_frame=9, language="old")
    model.seen = []

    def predict(obs):
        model.seen.append(obs)
        return np.ones((24, 8), np.float32), None, 0.1

    model._predict = predict
    return model


def values(model):
    return model.seen[-1]["video.wrist_image_left"][:, 0, 0, 0].tolist()


def test_anchor_then_newest_four_and_state(model):
    assert model.generate(request((1, 2))).chunk_index == 0
    assert values(model) == [2]
    assert model.generate(request((3,))).chunk_index == 1
    assert values(model) == [2, 2, 2, 3]
    model.generate(request((4, 5, 6, 7, 8)))
    assert values(model) == [5, 6, 7, 8]
    np.testing.assert_allclose(model.seen[-1]["state.joint_position"], [[0.1] * 7])
    assert model.seen[-1]["state.gripper_position"].item() == 0.5


def test_prompt_change_and_reset_start_new_episode(model):
    model.generate(request())
    model.generate(request((2,), "put it down"))
    assert values(model) == [2]
    assert model.episode.chunk_count == 1
    assert model._action_head.current_start_frame == 0
    assert model._action_head.language is None
    model.reset()
    assert model.episode is None
    assert model.generate(request((3,))).chunk_index == 0


def test_input_arrays_are_owned_by_episode(model):
    first = request()
    model.generate(first)
    first.cameras["wrist"][0][:] = 99
    model.generate(request((2,)))
    assert values(model) == [1, 1, 1, 2]


def test_invalid_input_does_not_advance(model):
    model.generate(request())
    with pytest.raises(InvalidRequest, match="missing frames"):
        model.generate(DreamZeroInput("other", {}))
    assert model.episode.chunk_count == 1


def test_failed_prediction_leaves_reset_to_caller(model):
    def fail(obs):
        raise RuntimeError("inference failed")

    model.generate(request())
    episode = model.episode
    model._predict = fail
    with pytest.raises(RuntimeError):
        model.generate(request())
    assert model.episode is episode
    assert episode.chunk_count == 1
    model.reset()
    assert model.episode is None


def test_model_has_no_runtime_dependency():
    tree = ast.parse(
        Path(__file__).parents[1].joinpath("dreamzero_model.py").read_text()
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("reactor_runtime")
        elif isinstance(node, ast.Import):
            assert all(
                not name.name.startswith("reactor_runtime") for name in node.names
            )


def test_resize_happens_once_when_frame_enters_history(model, monkeypatch):
    from unittest.mock import Mock

    import cv2

    resize = Mock(wraps=cv2.resize)
    monkeypatch.setattr(cv2, "resize", resize)
    large = np.ones((4, 6, 3), np.uint8)
    req = DreamZeroInput("pick", {name: (large,) for name in CAMERAS})
    model.generate(req)
    model.generate(req)
    assert resize.call_count == 6  # Three new frames per call, not the retained window.
    assert model.seen[-1]["video.wrist_image_left"].shape == (4, 2, 3, 3)


def test_nonleading_rank_advances_episode_without_output(model):
    model.rank = 1
    model._predict = lambda obs: (None, None, 0.1)
    assert model.generate(request()) is None
    assert model.episode.chunk_count == 1


def test_warmup_includes_preview_decode_and_resets(model):
    from unittest.mock import Mock

    model._decode_preview = Mock(return_value=np.zeros((9, 352, 640, 3), np.uint8))
    model._warmup()
    assert model._decode_preview.call_count == 2
    assert len(model.seen) == 2
    assert model.episode is None


@pytest.mark.parametrize(
    "invalid",
    [
        DreamZeroInput(" ", {}),
        DreamZeroInput("pick", {}),
        DreamZeroInput(
            "pick", {name: (np.zeros((2, 3), np.uint8),) for name in CAMERAS}
        ),
        DreamZeroInput(
            "pick", {name: (np.zeros((2, 3, 3), np.float32),) for name in CAMERAS}
        ),
        DreamZeroInput("pick", request().cameras, (0.0,)),
    ],
)
def test_request_errors_are_distinct_and_preserve_episode(model, invalid):
    model.generate(request())
    episode = model.episode
    with pytest.raises(InvalidRequest):
        model.generate(invalid)
    assert model.episode is episode
    assert episode.chunk_count == 1


def test_preview_failure_does_not_reset_model(model):
    from dataclasses import replace
    from unittest.mock import Mock

    model.generate(request())
    episode = model.episode
    model._decode_preview = Mock(side_effect=RuntimeError("decode failed"))
    with pytest.raises(RuntimeError, match="decode failed"):
        model.generate(replace(request(), video_preview=True))
    assert model.episode is episode
    assert episode.chunk_count == 1
