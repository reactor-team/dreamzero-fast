# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Exercise the runtime's actual process boundary without a GPU checkpoint."""

import numpy as np
import pytest
from reactor_runtime.distributed import DistributedRunner

from dreamzero_model import DreamZeroInput, DreamZeroModel, InvalidRequest


class FakeCheckpoint(DreamZeroModel):
    def load(self):
        assert self.world_size == 2
        assert self.rank in (0, 1)
        self._copy_inputs = False
        self._image_height, self._image_width = 2, 3

    def _build_obs(self, frames, request):
        return frames

    def _predict(self, obs):
        if self.rank != 0:
            return None, None, 0.0
        return np.full((24, 8), obs["wrist"][0][0, 0, 0], np.float32), None, 0.0


def test_two_process_generate_and_reset():
    runner = DistributedRunner(FakeCheckpoint, world_size=2, init_process_group=False)
    try:
        runner.start()
        with pytest.raises(InvalidRequest, match="task prompt"):
            runner.generate(DreamZeroInput(" ", {}))
        frame = np.full((2, 3, 3), 7, np.uint8)
        request = DreamZeroInput(
            "pick", {name: (frame,) for name in ("exterior_1", "exterior_2", "wrist")}
        )
        assert runner.generate(request).chunk_index == 0
        frame[:] = 9
        result = runner.generate(request)
        assert result.chunk_index == 1
        np.testing.assert_array_equal(result.actions, np.full((24, 8), 7, np.float32))
        runner.reset()
        reset_result = runner.generate(request)
        assert reset_result.chunk_index == 0
        assert reset_result.actions[0, 0] == 9
    finally:
        runner.shutdown()
