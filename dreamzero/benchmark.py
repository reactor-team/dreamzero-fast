# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Measure model-step latency on a saved observation sequence or synthetic RGB frames."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from dreamzero_model import DreamZeroInput, DreamZeroModel, read_config

CAMERAS = ("exterior_1", "exterior_2", "wrist")


def check_application(model) -> dict:
    """Check real application hooks and buffers; no network transport is involved."""
    from reactor_runtime import ApplicationError, InputBuffer, InputFrame, StepOutcome

    from dreamzero import DreamZero
    from model_types import DreamZeroMedia, DreamZeroState

    model.reset()
    app = DreamZero()
    app.model = model
    app.state = DreamZeroState()
    app.on_session_started()
    app.media = DreamZeroMedia(**{name: InputBuffer() for name in CAMERAS})
    sent = []

    async def send(message):
        sent.append(message.to_wire_format())

    app.send = send
    frame = np.random.default_rng(0).integers(0, 256, (180, 320, 3), dtype=np.uint8)

    def step():
        for name in CAMERAS:
            getattr(app.media, name).push(InputFrame(frame))
        request = asyncio.run(app.process_input())
        result = app.generate(request)
        output = asyncio.run(app.process_output(StepOutcome(result=result)))
        assert result.actions.shape == (24, 8)
        assert np.isfinite(result.actions).all()
        metadata = json.loads(output.__metadata__["predicted_video"][0])
        np.testing.assert_array_equal(metadata["data"]["actions"], result.actions)
        return result, output

    try:
        asyncio.run(app.process_input())
    except ApplicationError:
        pass
    else:
        raise AssertionError("an empty prompt was accepted")
    app.state.prompt = "pick up the ball"
    result, output = step()
    assert result.chunk_index == 0
    assert output.predicted_video.shape == (1, 352, 640, 3)
    assert step()[0].chunk_index == 1
    app.state.prompt = "put the ball down"
    assert step()[0].chunk_index == 0
    asyncio.run(app.reset())
    assert app.state.prompt == "put the ball down"
    app.state.video_preview = True
    result, output = step()
    assert result.chunk_index == 0
    assert output.predicted_video.dtype == np.uint8
    assert output.predicted_video.shape[1:] == (352, 640, 3)
    assert output.predicted_video.any()
    assert sum(message["type"] == "episode_started" for message in sent) == 3
    app.on_session_ended()
    return {
        "passed": True,
        "preview_frames": len(output.predicted_video),
        "checks": [
            "empty prompt refusal",
            "action metadata",
            "chunk progression",
            "prompt change",
            "explicit reset",
            "decoded preview",
            "session reset",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument(
        "--observations",
        type=Path,
        help="NPZ with RGB arrays (T,H,W,3) named for each camera",
    )
    parser.add_argument("--prompt", default="pick up the ball")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=0.0,
        help="Wait between steps, as a robot does while it executes a chunk",
    )
    parser.add_argument(
        "--check-app", action="store_true", help="Check application hooks after timing"
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 2:
        parser.error("use at least one warmup and two measured steps")
    config = read_config(args.config)
    if args.observations:
        with np.load(args.observations, allow_pickle=False) as data:
            cameras = {name: data[name].copy() for name in CAMERAS}
        input_hash = hashlib.sha256(args.observations.read_bytes()).hexdigest()
    else:
        rng = np.random.default_rng(0)
        cameras = {
            name: rng.integers(0, 256, (4, 180, 320, 3), dtype=np.uint8)
            for name in CAMERAS
        }
        input_hash = "synthetic-random-rgb-seed-0"
    if any(
        array.ndim != 4 or array.shape[-1] != 3 or len(array) == 0
        for array in cameras.values()
    ):
        parser.error("each camera must have a nonempty (T,H,W,3) array")
    import torch
    from reactor_runtime.distributed import DistributedRunner

    count = int(config.get("num_gpus", 1))
    kwargs = {
        "weights_path": str(args.weights.resolve()),
        "copy_inputs": False,
        "config_path": str(args.config.resolve()),
    }
    model = (
        DreamZeroModel()
        if count == 1
        else DistributedRunner(
            DreamZeroModel,
            world_size=count,
            load_kwargs=kwargs,
            call_timeout=120,
        )
    )
    records = []
    load_start = time.perf_counter()
    try:
        if count == 1:
            model.load(**kwargs)
        else:
            model.start()
        load_seconds = time.perf_counter() - load_start
        model.reset()
        for i in range(args.warmup + args.steps):
            request = DreamZeroInput(
                args.prompt,
                {name: (array[i % len(array)],) for name, array in cameras.items()},
                (0.0,) * 7,
                0.0,
                False,
            )
            start = time.perf_counter()
            result = model.generate(request)
            elapsed = time.perf_counter() - start
            if not np.isfinite(result.actions).all() or result.actions.shape[1:] != (
                8,
            ):
                raise ValueError("model returned invalid actions")
            row = {
                "chunk_index": result.chunk_index,
                "warmup": i < args.warmup,
                "generate_ms": elapsed * 1000,
                "prediction_ms": result.inference_seconds * 1000,
                "action_sha256": hashlib.sha256(result.actions.tobytes()).hexdigest(),
                "actions": result.actions.tolist(),
            }
            records.append(row)
            print(json.dumps(row), flush=True)
            if args.interval_ms > 0:
                time.sleep(args.interval_ms / 1000)
        measured = [row["generate_ms"] for row in records if not row["warmup"]]
        report = {
            "scope": "model.generate wall time; includes runner IPC for two GPUs; excludes transport and preview decode",
            "input": input_hash,
            "prompt": args.prompt,
            "config": config,
            "load_seconds": load_seconds,
            "warmup_steps": args.warmup,
            "measured_steps": args.steps,
            "interval_ms": args.interval_ms,
            "p50_ms": float(np.percentile(measured, 50)),
            "p95_ms": float(np.percentile(measured, 95)),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "runtime": version("reactor-runtime"),
            "tensorrt": version("tensorrt-cu12"),
            "model_source_sha256": hashlib.sha256(
                Path(__file__).with_name("dreamzero_model.py").read_bytes()
            ).hexdigest(),
            "checkpoint_config_sha256": hashlib.sha256(
                (args.weights / "checkpoint/config.json").read_bytes()
            ).hexdigest(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(count)],
            "samples": records,
        }
        if args.check_app:
            repeated = []
            for _ in range(2):
                model.reset()
                repeated.append(model.generate(request).actions.copy())
            report["reset_repeatability"] = {
                "exact_match": bool(np.array_equal(*repeated)),
                "max_absolute_difference": float(
                    np.max(np.abs(repeated[0] - repeated[1]))
                ),
                "actions": [actions.tolist() for actions in repeated],
            }
            report["application_checks"] = check_application(model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if isinstance(model, DistributedRunner):
            model.shutdown()


if __name__ == "__main__":
    main()
