# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Check load configuration with fake CUDA and checkpoint dependencies."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dreamzero_model import DreamZeroModel


@pytest.mark.parametrize("rank,size", [(0, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize("trt", [False, True])
def test_load_uses_assigned_device_and_topology(monkeypatch, tmp_path, rank, size, trt):
    model = DreamZeroModel()
    model.rank, model.world_size, model.device = rank, size, f"cuda:{rank + 3}"
    dist = SimpleNamespace(
        is_initialized=lambda: size == 2,
        init_process_group=Mock(),
        HashStore=Mock(),
        get_rank=lambda: rank,
    )
    mesh = Mock(return_value=object())
    torch = SimpleNamespace(
        distributed=dist, _dynamo=SimpleNamespace(config=SimpleNamespace())
    )
    head = SimpleNamespace(ip_rank=rank, ip_size=size)
    policy = Mock(
        return_value=SimpleNamespace(trained_model=SimpleNamespace(action_head=head))
    )
    for name, module in {
        "torch": torch,
        "torch.distributed": dist,
        "torch.distributed.device_mesh": SimpleNamespace(init_device_mesh=mesh),
        "tianshou.data": SimpleNamespace(Batch=object),
        "groot.vla.data.schema": SimpleNamespace(EmbodimentTag=lambda value: value),
        "groot.vla.model.n1_5.sim_policy": SimpleNamespace(GrootSimPolicy=policy),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    # Conflicting environment/config values must not override assigned topology.
    monkeypatch.setenv("LOCAL_RANK", "99")
    config = tmp_path / "config.yml"
    config.write_text(
        f"num_gpus: 99\nwarmup: false\nuse_trt_engine: {str(trt).lower()}\n"
    )
    engine = tmp_path / "checkpoint/tensorrt/wan/WanModel_nvfp4.trt"
    if trt:
        engine.parent.mkdir(parents=True)
        engine.touch()
    monkeypatch.setattr(model, "_local_umt5_tokenizer_dir", lambda _: None)
    model.load(tmp_path, config)
    assert policy.call_args.kwargs["device"] == f"cuda:{rank + 3}"
    mesh.assert_called_once_with("cuda", (size,), mesh_dim_names=("ip",))
    assert head.num_inference_steps == 16
    # The optimization settings default to the shipped behavior.
    assert head.prompt_cache is True
    assert head.vae_prefix is False
    if size == 1:
        dist.init_process_group.assert_called_once_with(
            "nccl", store=dist.HashStore.return_value, rank=0, world_size=1
        )
    else:
        dist.init_process_group.assert_not_called()
