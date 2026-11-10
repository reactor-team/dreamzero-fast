# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Exercise the policy wrapper without loading checkpoint dependencies."""

import ast
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("return_actions", [True, False])
def test_policy_only_preserves_state_and_leader_decodes(return_actions):
    source = Path(__file__).parents[1] / "groot/vla/model/n1_5/sim_policy.py"
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "lazy_joint_forward_causal"
    )
    namespace = {
        "np": np,
        "time": time,
        "torch": SimpleNamespace(is_tensor=lambda _: False, inference_mode=nullcontext),
        "Batch": SimpleNamespace,
    }
    # Compile only the checked-in method without importing GPU dependencies.
    exec(  # noqa: S102
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    class Camera(np.ndarray):
        def copy(self, *args, **kwargs):
            raise AssertionError("camera array must not be copied for action decoding")

    state = np.ones((1, 1, 7))
    obs = {
        "state.joint_position": state,
        "video.wrist": np.ones((1, 2, 3, 3)).view(Camera),
    }
    decoded = []

    def apply(batch):
        state[:] = 99  # The preserved state must survive transforms.
        batch.normalized_obs = {}
        return batch

    def unapply(batch, obs):
        decoded.append(obs)
        np.testing.assert_array_equal(obs["state.joint_position"], 1)
        assert set(obs) == {"state.joint_position"}
        return batch

    policy = SimpleNamespace(
        _check_state_is_batched=lambda _: True,
        apply=apply,
        unapply=unapply,
        eval_bf16=True,
        rank=0,
        trained_model=SimpleNamespace(
            lazy_joint_video_action_causal=lambda *args, **kwargs: {
                "action_pred": SimpleNamespace(float=lambda: "actions"),
                "video_pred": "latents",
            }
        ),
    )
    output, latents = namespace["lazy_joint_forward_causal"](
        policy, SimpleNamespace(obs=obs), return_actions=return_actions
    )
    assert latents == "latents"
    assert len(decoded) == int(return_actions)
    assert (output is not None) == return_actions
