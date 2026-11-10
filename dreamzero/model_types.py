# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Reactor type declarations for the DreamZero port.

Tracks, client-controllable state, and outbound messages for the
closed-loop DreamZero policy (see ``../docs/serving.md`` for the
client-facing contract).
"""

import numpy as np
from reactor_runtime import (
    InputField,
    InputState,
    MediaInput,
    ModelMessage,
    Output,
    Video,
)

# --------------------------------------------------------------------- #
#  Tracks                                                                #
# --------------------------------------------------------------------- #


class DreamZeroMedia(MediaInput):
    """Continuously streamed camera observations (DROID rig)."""

    exterior_1: Video
    exterior_2: Video
    wrist: Video


class DreamZeroOutput(Output):
    """Predicted-video preview track.

    A composite multi-view frame (wrist on top, the two exterior views
    on the bottom row). Emits black frames when ``video_preview`` is
    disabled so the track stays alive without paying for VAE decodes.
    """

    predicted_video: Video


# --------------------------------------------------------------------- #
#  Client state                                                          #
# --------------------------------------------------------------------- #


class DreamZeroState(InputState):
    # Frames received since the last step, at most four per camera.
    _pending: dict[str, list[np.ndarray]] | None = None

    prompt: str = InputField(
        default="",
        max_length=500,
        description="Natural-language task instruction. Changing it mid-episode "
        "re-anchors the model's causal cache on the latest observation.",
        moderate=True,
    )
    joint_position: list[float] = InputField(
        default=None,
        min_length=7,
        max_length=7,
        description="Measured robot joint positions (7 floats, radians). "
        "Snapshotted together with the camera frames each chunk. "
        "When unset, zeros are used and predicted joint targets are "
        "relative deltas rather than absolute positions.",
    )
    gripper_position: float = InputField(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Measured gripper position (0 = open, 1 = closed), "
        "snapshotted together with the camera frames each chunk.",
    )
    video_preview: bool = InputField(
        default=False,
        description="Decode the predicted-video latents to pixels on the "
        "predicted_video track. Costs one VAE decode per chunk; when off, "
        "black frames are emitted instead.",
    )


# --------------------------------------------------------------------- #
#  Outbound messages                                                     #
# --------------------------------------------------------------------- #


class EpisodeStarted(ModelMessage):
    """Sent with the first completed chunk of an episode."""

    prompt: str
    frames_per_chunk: int
    action_horizon: int


class ActionChunk(ModelMessage):
    """One predicted action chunk.

    ``actions`` is ``action_horizon x 8``: 7 joint positions plus 1
    gripper position per step. Joint values are absolute targets when
    the client streams its measured ``joint_position``; with the zero
    default they are relative deltas. The client decides when to execute or discard a chunk.

    """

    chunk_index: int
    actions: list  # (action_horizon, 8) nested list of floats
    inference_seconds: float


class EpisodeReset(ModelMessage):
    """The episode was cleared; prompt and robot state are unchanged."""
