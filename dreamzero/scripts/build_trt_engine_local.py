# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Rebuild the DreamZero nvfp4 TensorRT engine locally (x86_64 / B200).

The engine shipped in the checkpoint is a GB200 (aarch64) plan and TRT
plans are platform-locked, so it cannot load on an x86_64 host. This
rebuilds an equivalent engine from the shipped ONNX (which already carries
the nvfp4 FP4-QDQ nodes) using the same precision flags and dynamic-shape
profile as the checkpoint's build log (trtexec v101302).

Run inside the container (needs tensorrt-cu12 installed):

  docker run --rm --gpus device=0 -v "$PWD/weights:/weights" \
    --entrypoint python reactor-local/dreamzero:dev \
    scripts/build_trt_engine_local.py --weights /weights
"""

import argparse
import os
import time

import tensorrt as trt


# From WanModel_nvfp4_build.log: kv_cache_packed is the only dynamic input
# (axis 3 = kv cache length). 40 layers x 2 (k/v) x batch 1 x LEN x 40 heads
# x 128 head_dim.
KV_MIN = (40, 2, 1, 880, 40, 128)
KV_OPT = (40, 2, 1, 7920, 40, 128)
KV_MAX = (40, 2, 1, 8800, 40, 128)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--weights", default="/weights")
    args = cli.parse_args()
    wan_dir = os.path.join(args.weights, "checkpoint", "tensorrt", "wan")
    onnx_path = os.path.join(wan_dir, "CausalWanModel.onnx")
    engine_path = os.path.join(wan_dir, "WanModel_nvfp4.trt")
    assert os.path.exists(onnx_path), f"missing ONNX at {onnx_path}"
    print(f"tensorrt {trt.__version__}", flush=True)

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print("parsing ONNX (with external .onnx_data) ...", flush=True)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print("  parser error:", parser.get_error(i), flush=True)
        raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)
    # Precision flags matching the checkpoint build (--fp16 --bf16 --fp8);
    # FP4 comes from the ONNX QDQ nodes and, where the flag exists, is
    # enabled explicitly.
    for name in ("FP16", "BF16", "FP8", "FP4"):
        flag = getattr(trt.BuilderFlag, name, None)
        if flag is not None:
            config.set_flag(flag)
            print(f"  builder flag: {name}", flush=True)

    # Dynamic-shape profile for kv_cache_packed.
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        if inp.name == "kv_cache_packed":
            profile.set_shape(inp.name, KV_MIN, KV_OPT, KV_MAX)
            print(f"  profile kv_cache_packed: {KV_MIN} / {KV_OPT} / {KV_MAX}",
                  flush=True)
    config.add_optimization_profile(profile)

    print("building serialized engine (this can take many minutes) ...", flush=True)
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed")
    print(f"built in {time.perf_counter() - t0:.0f}s", flush=True)

    tmp = engine_path + ".x86.tmp"
    with open(tmp, "wb") as f:
        f.write(serialized)
    os.replace(tmp, engine_path)
    print(f"wrote engine -> {engine_path} ({os.path.getsize(engine_path) / 1e9:.2f} GB)",
          flush=True)


if __name__ == "__main__":
    main()
