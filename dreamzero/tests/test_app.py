# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from reactor_runtime import ApplicationError, StepOutcome

from dreamzero import CAMERAS, DreamZero
from dreamzero_model import DreamZeroResult
from model_types import DreamZeroState


class Track:
    def __init__(self):
        self.frames = []

    def try_read(self, count, mode):
        return [SimpleNamespace(data=self.frames.pop(0))] if self.frames else None


@pytest.fixture
def app():
    app = DreamZero()
    app.state = DreamZeroState(prompt="pick up the ball")
    app.media = SimpleNamespace(**{name: Track() for name in CAMERAS})
    app.on_session_started()
    app.model = Mock()
    app.send = AsyncMock()
    return app


def run(coro):
    return asyncio.run(coro)


def test_refusal_keeps_partial_camera_inputs(app):
    frame = np.zeros((2, 3, 3), np.uint8)
    app.media.wrist.frames.append(frame)
    with pytest.raises(ApplicationError, match="exterior_1"):
        run(app.process_input())
    for name in ("exterior_1", "exterior_2"):
        getattr(app.media, name).frames.append(frame)
    request = run(app.process_input())
    assert set(request.cameras) == set(CAMERAS)
    assert request.cameras["wrist"][0] is frame
    with pytest.raises(ApplicationError):
        run(app.process_input())


def test_empty_prompt_refuses_before_consuming(app):
    app.state.prompt = " "
    app.media.wrist.frames.append(np.zeros((2, 3, 3), np.uint8))
    with pytest.raises(ApplicationError, match="prompt"):
        run(app.process_input())
    assert len(app.media.wrist.frames) == 1


def test_output_contains_actions_for_streamed_and_stepped_clients(app):
    result = DreamZeroResult("pick", 0, np.ones((24, 8)), 0.1, None)
    output = run(app.process_output(StepOutcome(result=result)))
    assert app.send.await_count == 2
    assert output.predicted_video.shape == (1, 352, 640, 3)
    payload = json.loads(output.__metadata__["predicted_video"][0])
    assert payload["data"]["chunk_index"] == 0
    assert len(payload["data"]["actions"]) == 24


def test_error_is_not_hidden(app):
    with pytest.raises(RuntimeError, match="failed"):
        run(app.process_output(StepOutcome(error=RuntimeError("failed"))))
    app.send.assert_not_called()


def test_reset_keeps_state_and_drops_pending_frames(app):
    app.state._pending["wrist"] = [np.zeros((2, 3, 3), np.uint8)]
    run(app.reset())
    app.model.reset.assert_called_once()
    assert app.state.prompt == "pick up the ball"
    assert not any(app.state._pending.values())


def test_session_end_resets_model(app):
    app.on_session_ended()
    app.model.reset.assert_called_once()


def test_closed_camera_refuses_without_stale_pending_frames(app):
    from reactor_runtime import BufferClosed

    app.state._pending["wrist"] = [np.zeros((2, 3, 3), np.uint8)]
    app.media.wrist.try_read = Mock(side_effect=BufferClosed("closed"))
    with pytest.raises(ApplicationError, match="camera track closed: wrist"):
        run(app.process_input())
    assert not app.state._pending["wrist"]
    app.model.generate.assert_not_called()


def test_reset_and_session_end_flush_queued_output(app):
    app.output.flush = Mock()
    run(app.reset())
    app.output.flush.assert_called_once_with()
    app.on_session_ended()
    assert app.output.flush.call_count == 2


def test_frame_metadata_supports_flat_and_enveloped_actions(app):
    result = DreamZeroResult("pick", 0, np.ones((24, 8)), 0.1, None)
    output = run(app.process_output(StepOutcome(result=result)))
    payload = json.loads(output.__metadata__["predicted_video"][0])
    for name in ("actions", "chunk_index", "inference_seconds"):
        assert payload[name] == payload["data"][name]


def test_pending_frames_are_session_local(app):
    frame = np.zeros((2, 3, 3), np.uint8)
    app.state._pending["wrist"] = [frame]
    previous = app.state
    app.state = DreamZeroState(prompt="new session")
    assert app.state._pending is None
    app.on_session_started()
    assert not any(app.state._pending.values())
    assert previous._pending["wrist"][0] is frame
    assert "_pending" not in DreamZeroState._public_fields


def test_failure_cleanup_is_owned_by_session_end(app):
    error = ValueError("failed inside inference")
    with pytest.raises(ValueError) as raised:
        run(app.process_output(StepOutcome(error=error)))
    assert raised.value is error
    app.model.reset.assert_not_called()
    app.on_session_ended()
    app.model.reset.assert_called_once_with()


def test_reset_command_describes_retained_state():
    from reactor_runtime.interface.events.decorators import EVENT_ATTR

    command = getattr(DreamZero.reset, EVENT_ATTR)
    assert command.name == "reset"
    assert "prompt and robot state are kept" in command.description
