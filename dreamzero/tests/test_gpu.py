# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Opt-in checkpoint test of the application's real input/output hooks."""

import os
from pathlib import Path

import pytest
from reactor_runtime.distributed import DistributedRunner

from benchmark import check_application
from dreamzero import DreamZero

pytestmark = pytest.mark.skipif(
    not os.environ.get("DREAMZERO_TEST_WEIGHTS"),
    reason="requires a checkpoint and reserved GPUs",
)


def test_checkpoint_application(monkeypatch):
    monkeypatch.setenv("REACTOR_WEIGHTS_PATH", os.environ["DREAMZERO_TEST_WEIGHTS"])
    app = DreamZero()
    app.load(Path(os.environ.get("DREAMZERO_TEST_CONFIG", "config.yml")))
    try:
        assert check_application(app.model)["passed"]
    finally:
        if isinstance(app.model, DistributedRunner):
            app.model.shutdown()
