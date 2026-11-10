# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Opt-in SDK tests against a running DreamZero server with real weights."""

import asyncio
import contextlib
import json
import os
import time
import urllib.request
from pathlib import Path

import pytest
from policy_client import (
    CAMERAS,
    action_chunk,
    media_at,
    observations,
    result_chunk,
    stream_observations,
)
from reactor_sdk import Reactor, StepRefusedError

URL = os.environ.get("DREAMZERO_E2E_URL")
pytestmark = pytest.mark.skipif(not URL, reason="requires a running DreamZero server")


def session(mode):
    return Reactor(
        "dreamzero-fast",
        local=True,
        api_url=URL,
        generation=mode,
        step_timeout=60 if mode == "stepped" else None,
    )


async def connect(client):
    def health():
        with urllib.request.urlopen(f"{URL}/health", timeout=5) as response:
            return json.load(response)["state"]

    # Disconnect completes before the resident model finishes session cleanup.
    async with asyncio.timeout(30):
        while await asyncio.to_thread(health) != "available":
            await asyncio.sleep(0.1)
    await client.connect()
    for name in CAMERAS:
        await client.track(name).publish()


def save_report(name, report):
    folder = os.environ.get("DREAMZERO_E2E_RESULTS")
    if folder:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")


def test_stepped_session():
    async def check():
        views = observations(None)
        report = {"steps": [], "checks": []}
        async with session("stepped") as client:
            await connect(client)
            try:
                refused = await client.step(state={"prompt": "pick up the ball"})
            except StepRefusedError:
                pass
            else:
                assert refused.error is not None
            report["checks"].append("missing-camera refusal")

            async def step(index, *, state=None, commands=None, expected_index=None):
                started = time.perf_counter()
                result = await client.step(
                    state=state, commands=commands, media=media_at(views, index)
                )
                chunk = result_chunk(result)
                if expected_index is not None:
                    assert chunk["chunk_index"] == expected_index
                frames = result.track("predicted_video").frames
                for frame in frames:
                    metadata = json.loads(frame.metadata)
                    assert metadata["actions"] == metadata["data"]["actions"]
                    assert action_chunk(metadata)["chunk_index"] == chunk["chunk_index"]
                    assert frame.array.shape == (352, 640, 3)
                report["steps"].append(
                    {
                        "chunk_index": chunk["chunk_index"],
                        "round_trip_ms": (time.perf_counter() - started) * 1000,
                        "policy_ms": chunk["inference_seconds"] * 1000,
                        "frames": len(frames),
                    }
                )
                return frames

            for index in range(5):
                frames = await step(
                    index,
                    state={"joint_position": [0.0] * 7, "gripper_position": 0.0},
                    expected_index=index,
                )
                assert len(frames) == 1 and not frames[0].array.any()
            report["checks"].append("chunk progression and both metadata forms")
            await step(0, state={"prompt": "move the ball"}, expected_index=0)
            report["checks"].append("prompt change")
            frames = await step(
                1,
                state={"prompt": "put down the ball", "video_preview": True},
                commands=[{"command": "reset"}],
                expected_index=0,
            )
            assert len(frames) > 1 and any(frame.array.any() for frame in frames)
            report["checks"].append("state plus reset in one step and decoded preview")
            await step(2, state={"video_preview": False}, expected_index=1)
        # Reconnect to the same resident model; history must not survive sessions.
        async with session("stepped") as client:
            await connect(client)
            result = await client.step(
                state={"prompt": "pick up the ball"}, media=media_at(views, 0)
            )
            assert result_chunk(result)["chunk_index"] == 0
        report["checks"].append("new session starts at chunk zero")
        save_report("stepped", report)

    asyncio.run(check())


def test_streamed_session():
    async def check():
        views = observations(None)
        messages = asyncio.Queue()
        loop = asyncio.get_running_loop()
        report = {"chunks": [], "checks": []}
        async with session("streamed") as client:

            def receive(message):
                if message.get("type") == "action_chunk":
                    loop.call_soon_threadsafe(messages.put_nowait, message)

            client.on("message", receive)
            await connect(client)
            await client.send_command("set_prompt", {"prompt": "pick up the ball"})
            await client.send_command(
                "set_joint_position", {"joint_position": [0.0] * 7}
            )
            publisher = asyncio.create_task(stream_observations(client, views))
            try:
                for index in range(8):
                    chunk = action_chunk(await asyncio.wait_for(messages.get(), 30))
                    assert chunk["chunk_index"] == index
                    report["chunks"].append(chunk["chunk_index"])
                await client.send_command("reset", {})
                report["reset_observed"] = []
                previous = 7
                async with asyncio.timeout(30):
                    while True:
                        chunk = action_chunk(await messages.get())
                        index = chunk["chunk_index"]
                        report["reset_observed"].append(index)
                        if index == 0:
                            break
                        # Messages sent before the reset acknowledgement can
                        # already be queued at the client.
                        assert index == previous + 1
                        previous = index
                report["checks"] = [
                    "streamed camera inputs and action messages",
                    "reset",
                ]
            finally:
                publisher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await publisher
        save_report("streamed", report)

    asyncio.run(check())
