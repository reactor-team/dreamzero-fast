# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""DreamZero checkpoint and episode state, independent of the serving runtime."""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)
FRAMES_PER_CHUNK = 4
_CAMERA_TO_MODEL_KEY = {
    "exterior_1": "video.exterior_image_1_left",
    "exterior_2": "video.exterior_image_2_left",
    "wrist": "video.wrist_image_left",
}


def read_config(path: Path | None) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text()) if path else {}
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError("config must be a YAML mapping")
    return config


class InvalidRequest(ValueError):
    """The request has missing or malformed observations or robot state."""


@dataclass(frozen=True)
class DreamZeroInput:
    prompt: str
    cameras: dict[str, tuple[np.ndarray, ...]]
    joint_position: tuple[float, ...] | None = None
    gripper_position: float = 0.0
    video_preview: bool = False


@dataclass(frozen=True)
class DreamZeroResult:
    prompt: str
    chunk_index: int
    actions: np.ndarray
    inference_seconds: float
    predicted_video: np.ndarray | None


@dataclass
class Episode:
    prompt: str
    chunk_count: int = 0
    frames: dict[str, deque] = field(
        default_factory=lambda: {
            name: deque(maxlen=FRAMES_PER_CHUNK) for name in _CAMERA_TO_MODEL_KEY
        }
    )


class DreamZeroModel:
    def __init__(self) -> None:
        self.rank = 0
        self.world_size = 1
        self.device = "cuda"
        self.episode: Episode | None = None
        self._copy_inputs = True

    def load(
        self,
        weights_path: str | Path,
        config_path: str | Path | None = None,
        *,
        copy_inputs: bool = True,
    ) -> None:
        # Runtime inputs already own their buffers. Direct callers may reuse
        # their arrays, so copying remains the standalone default.
        self._copy_inputs = copy_inputs
        config = read_config(Path(config_path) if config_path else None)
        weights = Path(weights_path)
        os.environ["HF_HOME"] = str(weights / "hf")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["ATTENTION_BACKEND"] = str(config.get("attention_backend", "torch"))
        os.environ["NUM_DIT_STEPS"] = str(int(config.get("num_dit_steps", 16)))
        os.environ["DYNAMIC_CACHE_SCHEDULE"] = str(
            bool(config.get("dynamic_cache_schedule", False))
        ).lower()
        os.environ.pop("LOAD_TRT_ENGINE", None)
        engine = weights / "checkpoint/tensorrt/wan/WanModel_nvfp4.trt"
        if config.get("use_trt_engine", False):
            if not engine.is_file():
                raise FileNotFoundError(
                    f"TensorRT was requested but the engine is missing: {engine}"
                )
            os.environ["LOAD_TRT_ENGINE"] = str(engine)

        import torch
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh

        if self.world_size not in (1, 2):
            raise ValueError("DreamZero requires one or two GPUs")
        if not dist.is_initialized():
            if self.world_size != 1:
                raise RuntimeError("two GPUs require an initialized distributed group")
            dist.init_process_group(
                "nccl", store=dist.HashStore(), rank=0, world_size=1
            )
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("ip",))
        torch._dynamo.config.recompile_limit = 800
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = 1000
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = 2000

        from tianshou.data import Batch

        from groot.vla.data.schema import EmbodimentTag
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

        self._torch, self._Batch = torch, Batch
        self._policy = GrootSimPolicy(
            embodiment_tag=EmbodimentTag(config.get("embodiment_tag", "oxe_droid")),
            model_path=str(weights / "checkpoint"),
            tokenizer_path_override=self._local_umt5_tokenizer_dir(weights),
            device=self.device,
            device_mesh=mesh,
        )
        self._action_head = self._policy.trained_model.action_head
        if (self._action_head.ip_rank, self._action_head.ip_size) != (
            dist.get_rank(),
            self.world_size,
        ):
            raise RuntimeError(
                "the policy did not initialize classifier-free guidance correctly"
            )
        steps = int(config.get("num_inference_steps", 16))
        if steps < 1:
            raise ValueError("num_inference_steps must be positive")
        self._action_head.num_inference_steps = steps
        self._action_head.prompt_cache = bool(config.get("prompt_cache", True))
        self._action_head.vae_prefix = bool(config.get("vae_prefix", False))
        self._image_height, self._image_width = self._expected_video_resolution(
            int(config.get("image_height", 0)), int(config.get("image_width", 0))
        )
        self.reset()
        if config.get("warmup", True):
            self._warmup()

    def reset(self) -> None:
        self.episode = None
        head = getattr(self, "_action_head", None)
        if head is not None:
            head.current_start_frame = 0
            head.language = None

    def generate(self, request: DreamZeroInput) -> DreamZeroResult | None:
        if not request.prompt.strip():
            raise InvalidRequest("a task prompt is required")
        for name in _CAMERA_TO_MODEL_KEY:
            frames = request.cameras.get(name, ())
            if not frames:
                raise InvalidRequest(f"missing frames for {name}")
            if any(
                f.ndim != 3 or f.shape[-1] != 3 or f.dtype != np.uint8 for f in frames
            ):
                raise InvalidRequest(f"{name} must contain uint8 RGB frames")
        if request.joint_position is not None and len(request.joint_position) != 7:
            raise InvalidRequest("joint_position must contain seven values")
        if self.episode is None or self.episode.prompt != request.prompt:
            self.reset()
            self.episode = Episode(request.prompt)
        episode = self.episode
        for name, window in episode.frames.items():
            incoming = request.cameras[name]
            # Frames older than the episode anchor must not reappear later.
            newest = (
                incoming[-1:]
                if episode.chunk_count == 0
                else incoming[-FRAMES_PER_CHUNK:]
            )
            window.extend(self._prepare_frame(frame) for frame in newest)
        frames = {
            name: [window[-1]]
            if episode.chunk_count == 0
            else self._pad_oldest(list(window), FRAMES_PER_CHUNK)
            for name, window in episode.frames.items()
        }
        actions, latents, elapsed = self._predict(self._build_obs(frames, request))
        video = (
            self._decode_preview(latents)
            if request.video_preview and self.rank == 0
            else None
        )
        chunk_index = episode.chunk_count
        episode.chunk_count += 1
        if actions is None:
            return None
        return DreamZeroResult(request.prompt, chunk_index, actions, elapsed, video)

    def _predict(self, obs: dict) -> tuple[np.ndarray | None, Any, float]:
        torch = self._torch
        # A deferred conditioning encode from the last chunk must be queued
        # before this chunk uses the GPU.
        self._action_head.join_deferred_ys()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            result, latents = self._policy.lazy_joint_forward_causal(
                self._Batch(obs=obs), return_actions=self.rank == 0
            )
        # The runtime returns only rank zero's result. The other rank must
        # finish its collective work, but need not copy actions to the CPU.
        actions = None
        if self.rank == 0:
            actions = self._convert_action(
                {
                    name: getattr(result.act, name)
                    for name in dir(result.act)
                    if name.startswith("action.")
                }
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        # With vae_prefix, finish the conditioning encode on a side stream
        # while the robot executes this chunk.
        self._action_head.start_deferred_ys()
        return actions, latents, elapsed

    def _warmup(self) -> None:
        frame = np.zeros((self._image_height, self._image_width, 3), dtype=np.uint8)
        request = DreamZeroInput(
            "warmup",
            {name: (frame,) for name in _CAMERA_TO_MODEL_KEY},
            video_preview=True,
        )
        try:
            self.generate(request)
            self.generate(request)
        finally:
            self.reset()

    def _decode_preview(self, latents: Any) -> np.ndarray | None:
        if latents is None:
            return None
        torch, head = self._torch, self._action_head
        with torch.no_grad():
            decoded = head.vae.decode(
                latents,
                tiled=head.tiled,
                tile_size=(head.tile_size_height, head.tile_size_width),
                tile_stride=(head.tile_stride_height, head.tile_stride_width),
            )
        return (
            ((decoded[0].permute(1, 2, 3, 0).float() + 1) * 127.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

    @staticmethod
    def _local_umt5_tokenizer_dir(weights) -> str | None:
        """Return the bundled google/umt5-xxl tokenizer snapshot, if present."""
        import glob

        snaps = glob.glob(
            str(weights / "hf" / "hub" / "models--google--umt5-xxl" / "snapshots" / "*")
        )
        return snaps[0] if snaps else None

    def _expected_video_resolution(self, cfg_h: int, cfg_w: int) -> tuple[int, int]:
        """(H, W) the checkpoint's eval transform expects for incoming video."""
        if cfg_h and cfg_w:
            return cfg_h, cfg_w
        eval_transform = getattr(self._policy, "eval_transform", None)
        if eval_transform is not None:
            for t in getattr(eval_transform, "transforms", []):
                res = getattr(t, "original_resolutions", None)
                if res:
                    w, h = next(iter(res.values()))
                    return int(h), int(w)
        return 180, 320

    @staticmethod
    def _pad_oldest(window: list[np.ndarray], n: int) -> list[np.ndarray]:
        """Front-pad to *n* frames by repeating the oldest, like the
        reference client does before enough frames exist."""
        padded = list(window)
        while len(padded) < n:
            padded.insert(0, padded[0])
        return padded

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] != (self._image_height, self._image_width):
            import cv2

            # Resize only newly received frames, before retaining the window.
            return cv2.resize(
                frame,
                (self._image_width, self._image_height),
                interpolation=cv2.INTER_LINEAR,
            )
        return frame.copy() if self._copy_inputs else frame

    def _build_obs(
        self, frames: dict[str, list[np.ndarray]], request: DreamZeroInput
    ) -> dict:
        """Assemble the model observation from paired frames + state."""
        obs: dict[str, Any] = {
            _CAMERA_TO_MODEL_KEY[name]: np.stack(frame_list, axis=0)
            for name, frame_list in frames.items()
        }

        # State is snapshotted at the same moment as the frame pairing.
        joint = np.asarray(
            request.joint_position if request.joint_position is not None else [0.0] * 7,
            dtype=np.float64,
        )
        gripper = np.asarray([request.gripper_position], dtype=np.float64)
        obs["state.joint_position"] = joint.reshape(1, -1)
        obs["state.gripper_position"] = gripper.reshape(1, -1)
        obs["annotation.language.action_text"] = request.prompt
        return obs

    @staticmethod
    def _convert_action(action_dict: dict) -> np.ndarray:
        """Convert the model action dict to an (N, 8) array (7 joints + gripper)."""
        import torch

        joint_action = None
        gripper_action = None
        for key, value in action_dict.items():
            if ("joint_position" in key or "joint_pos" in key) and "gripper" not in key:
                joint_action = value
            elif "gripper" in key:
                gripper_action = value
        if joint_action is None:
            raise ValueError("policy output has no joint actions")
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        n = joint_action.shape[0]
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            if gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
            elif gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            if gripper_action.shape[-1] > 1:
                gripper_action = gripper_action[..., :1]
        else:
            gripper_action = np.zeros((n, 1), dtype=np.float32)
        return np.concatenate([joint_action, gripper_action], axis=-1).astype(
            np.float32
        )
