# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
# /// script
# requires-python = ">=3.12"
# dependencies = ["reactor-sdk==1.13.0", "numpy>=2.1"]
# ///
"""Send saved or synthetic observations to DreamZero and print action chunks."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from reactor_sdk import Reactor, time_micros

CAMERAS = ("exterior_1", "exterior_2", "wrist")


def action_chunk(metadata: bytes | str | dict) -> dict[str, Any]:
    """Accept flat action fields, a message envelope, or both forms together."""
    payload = json.loads(metadata) if isinstance(metadata, (str, bytes)) else metadata
    if not isinstance(payload, dict):
        raise TypeError("action metadata must be an object")
    chunk = payload.get("data", payload)
    if not isinstance(chunk, dict) or "actions" not in chunk:
        raise ValueError("action metadata has no actions")
    if "data" in payload and "actions" in payload:
        for key in ("actions", "chunk_index", "inference_seconds"):
            if payload.get(key) != chunk.get(key):
                raise ValueError(f"conflicting action metadata: {key}")
    actions = np.asarray(chunk["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 8:
        raise ValueError("actions must have shape (N, 8)")
    if not np.isfinite(actions).all():
        raise ValueError("actions contain nonfinite values")
    index = chunk.get("chunk_index")
    # Protobuf Struct transports message numbers as doubles.
    if (
        isinstance(index, bool)
        or not isinstance(index, (int, float))
        or not np.isfinite(index)
        or index < 0
        or int(index) != index
    ):
        raise ValueError("invalid chunk index")
    return {**chunk, "chunk_index": int(index)}


def observations(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        rng = np.random.default_rng(0)
        return {
            name: rng.integers(0, 256, (4, 180, 320, 3), dtype=np.uint8)
            for name in CAMERAS
        }
    with np.load(path, allow_pickle=False) as saved:
        result = {name: saved[name].copy() for name in CAMERAS}
    for name, frames in result.items():
        if frames.ndim != 4 or frames.shape[-1] != 3 or not len(frames):
            raise ValueError(f"{name} must have shape (T, H, W, 3)")
        if frames.dtype != np.uint8:
            raise ValueError(f"{name} must contain uint8 RGB pixels")
    return result


def media_at(views: dict[str, np.ndarray], index: int) -> dict[str, list[np.ndarray]]:
    return {name: [frames[index % len(frames)]] for name, frames in views.items()}


async def stream_observations(session, views, fps=10.0):
    """Publish camera frames continuously; streamed media is not request/reply."""
    index = 0
    while True:
        captured = time_micros()
        for name, frames in media_at(views, index).items():
            session.track(name).push_frame(frames[0], capture_time_us=captured)
        index += 1
        await asyncio.sleep(1.0 / fps)


def result_chunk(result) -> dict[str, Any]:
    if result.error is not None:
        raise RuntimeError(f"step failed: {result.error}")
    frames = result.track("predicted_video").frames
    if not frames or frames[0].metadata is None:
        raise ValueError("step returned no action metadata")
    return action_chunk(frames[0].metadata)


async def run(args: argparse.Namespace) -> None:
    views = observations(args.observations)
    state = {
        "prompt": args.prompt,
        "joint_position": args.joints,
        "gripper_position": args.gripper,
        "video_preview": args.preview,
    }
    async with Reactor(
        args.model,
        api_key=os.environ.get("REACTOR_API_KEY"),
        local=args.local,
        generation=args.mode,
        **({"api_url": args.url} if args.url else {}),
    ) as session:
        received: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_message(message):
            if message.get("type") == "action_chunk":
                loop.call_soon_threadsafe(received.put_nowait, message)

        session.on("message", on_message)
        await session.connect()
        for name in CAMERAS:
            await session.track(name).publish()
        if args.mode == "stepped":
            for index in range(args.steps):
                result = await session.step(
                    state=state if index == 0 else None,
                    media=media_at(views, index),
                )
                print(json.dumps(result_chunk(result)), flush=True)
        else:
            for name, value in state.items():
                await session.send_command(f"set_{name}", {name: value})
            publisher = asyncio.create_task(
                stream_observations(session, views, args.fps)
            )
            try:
                for _ in range(args.steps):
                    message = await asyncio.wait_for(received.get(), timeout=30)
                    print(json.dumps(action_chunk(message)), flush=True)
            finally:
                publisher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await publisher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dreamzero-fast")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--url", help="local runtime URL or hosted API URL")
    parser.add_argument("--mode", choices=("stepped", "streamed"), default="stepped")
    parser.add_argument(
        "--observations", type=Path, help="NPZ containing the three RGB camera arrays"
    )
    parser.add_argument("--prompt", default="pick up the ball")
    parser.add_argument("--joints", type=float, nargs=7, default=[0.0] * 7)
    parser.add_argument("--gripper", type=float, default=0.0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--fps", type=float, default=10.0, help="camera replay rate in streamed mode"
    )
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.gripper <= 1 or not 0 < args.fps <= 120:
        parser.error("steps must be positive, gripper in [0, 1], and fps in (0, 120]")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
