# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Check the causal property that the VAE prefix depends on."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("tqdm")

from groot.vla.model.dreamzero.modules.wan_video_vae import VideoVAE_


@torch.no_grad()
def test_prefix_latents_match_full_encode():
    torch.manual_seed(0)
    vae = VideoVAE_(dim=8, z_dim=4).eval()
    scale = [torch.randn(4), torch.rand(4) + 0.5]
    image = torch.randn(1, 3, 1, 32, 32)
    clip = torch.cat([image, torch.zeros(1, 3, 32, 32, 32)], dim=2)

    full = vae.encode(clip, scale)
    prefix = vae.encode(clip[:, :, :9], scale)

    assert full.shape == (1, 4, 9, 4, 4)
    # Latent frames 0-2 depend only on input frames 0-8.
    assert torch.equal(prefix, full[:, :, :3])
    # The later latents depend on the camera frame, so a constant cannot
    # replace them. The model computes them from a full encode.
    other = vae.encode(torch.cat([image + 1, clip[:, :, 1:]], dim=2), scale)
    assert not torch.equal(other[:, :, 3:], full[:, :, 3:])


@torch.no_grad()
def test_encode_groups_matches_encode():
    torch.manual_seed(0)
    vae = VideoVAE_(dim=8, z_dim=4).eval()
    scale = [torch.randn(4), torch.rand(4) + 0.5]
    clip = torch.cat(
        [torch.randn(1, 3, 1, 32, 32), torch.zeros(1, 3, 8, 32, 32)], dim=2
    )
    cache = [None] * vae._enc_conv_num
    assert torch.equal(
        vae.encode_groups(clip, scale, cache, first=True), vae.encode(clip, scale)
    )
