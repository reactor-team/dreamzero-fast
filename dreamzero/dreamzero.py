# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Serve DreamZero camera observations and return robot action chunks."""

import json
from pathlib import Path

import numpy as np
from reactor_runtime import (
    ApplicationError,
    BufferClosed,
    ReactorApp,
    ReadMode,
    StepOutcome,
    TrackPayload,
    event,
    get_weights_path,
    session_ended,
    session_started,
)
from reactor_runtime.distributed import DistributedRunner

from dreamzero_model import DreamZeroInput, DreamZeroModel, DreamZeroResult, read_config
from model_types import (
    ActionChunk,
    DreamZeroMedia,
    DreamZeroOutput,
    DreamZeroState,
    EpisodeReset,
    EpisodeStarted,
)
from observation_align import AlignmentError, StampedFrame, align_pending

CAMERAS = ("exterior_1", "exterior_2", "wrist")


class DreamZero(ReactorApp):
    """Predict joint and gripper actions from three camera views and robot state."""

    media: DreamZeroMedia
    state: DreamZeroState
    buffer_size = 16

    def load(self, config_path: Path | None) -> None:
        config = read_config(config_path)
        count = int(config.get("num_gpus", 1))
        if count not in (1, 2):
            raise ValueError("num_gpus must be 1 or 2")
        kwargs = {
            "weights_path": str(get_weights_path()),
            "config_path": config_path,
            "copy_inputs": False,
        }
        if count == 1:
            self.model = DreamZeroModel()
            self.model.load(**kwargs)
        else:
            self.model = DistributedRunner(
                DreamZeroModel,
                world_size=count,
                load_kwargs=kwargs,
                call_timeout=120,
                start_timeout=3600,
            )
            self.model.start()

    @session_started
    def on_session_started(self) -> None:
        self.state._pending = {name: [] for name in CAMERAS}

    async def process_input(self) -> DreamZeroInput:
        if not self.state.prompt.strip():
            raise ApplicationError("set a task prompt before requesting actions")
        # Keep frames gathered during a refusal: cameras can arrive separately.
        for name in CAMERAS:
            track = getattr(self.media, name)
            try:
                while (frames := track.try_read(1, mode=ReadMode.FIFO)) is not None:
                    for frame in frames:
                        self.state._pending[name].append(
                            StampedFrame(frame.data, frame.capture_time_us)
                        )
                    self.state._pending[name] = self.state._pending[name][-4:]
            except BufferClosed:
                self.state._pending[name] = []
                raise ApplicationError(f"camera track closed: {name}") from None
        try:
            cameras = align_pending(self.state._pending, cameras=CAMERAS)
        except AlignmentError as err:
            raise ApplicationError(str(err)) from None
        # Snapshot robot state only once cameras share a capture moment.
        request = DreamZeroInput(
            prompt=self.state.prompt,
            cameras=cameras,
            joint_position=tuple(self.state.joint_position)
            if self.state.joint_position is not None
            else None,
            gripper_position=self.state.gripper_position,
            video_preview=self.state.video_preview,
        )
        return request

    def generate(self, request: DreamZeroInput) -> DreamZeroResult:
        return self.model.generate(request)

    async def process_output(self, outcome: StepOutcome) -> DreamZeroOutput:
        if outcome.error is not None:
            # Inference may have partially advanced the episode or failed on a GPU.
            # Continuing would reuse uncertain state, so end the session and let
            # its teardown reset the model instead of attempting recovery here.
            raise outcome.error
        result: DreamZeroResult = outcome.result
        if result.chunk_index == 0:
            await self.send(
                EpisodeStarted(
                    prompt=result.prompt,
                    frames_per_chunk=4,
                    action_horizon=len(result.actions),
                )
            )
        chunk = ActionChunk(
            chunk_index=result.chunk_index,
            actions=result.actions.tolist(),
            inference_seconds=result.inference_seconds,
        )
        await self.send(chunk)
        video = result.predicted_video
        if video is None:
            video = np.zeros((1, 352, 640, 3), dtype=np.uint8)
        # A stepped response carries media. Put actions on each frame as well
        # as sending the message used by streaming clients.
        payload = chunk.to_wire_format()
        # Support clients reading either the message envelope or flat fields.
        payload.update(payload["data"])
        metadata = json.dumps(payload).encode()
        return DreamZeroOutput(
            predicted_video=TrackPayload(
                video,
                metadata=[metadata] * len(video),
            )
        )

    @event(
        name="reset",
        description="End the episode. The next step starts from its own observation; "
        "the prompt and robot state are kept.",
    )
    async def reset(self) -> EpisodeReset:
        """Clear the episode while keeping the client's prompt and robot state."""
        self.model.reset()
        self.state._pending = {name: [] for name in CAMERAS}
        self.output.flush()
        return EpisodeReset()

    @session_ended
    def on_session_ended(self) -> None:
        self.model.reset()
        self.output.flush()
