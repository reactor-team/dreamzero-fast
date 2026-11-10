# Measuring step latency

Run the benchmark inside the model environment, with the weights mounted and the selected GPUs visible:

```sh
cd dreamzero
python benchmark.py --weights /weights --config config.yml \
  --warmup 5 --steps 50 --check-app --output results.json
```

By default it uses seeded synthetic RGB images, zero joint state and previews off. This checks inference timing and finite action outputs. It does not measure robot behavior or task success. Pass `--observations frames.npz` to replay saved RGB arrays named `exterior_1`, `exterior_2` and `wrist`, each shaped `(T,H,W,3)`. The replay cycles through the arrays; it does not close a simulator control loop.

The JSON records the configuration, input identity, software versions, GPU models, load time and every timing sample. Warmup samples remain in the file but are excluded from the reported median and p95. Each sample also includes the returned actions and their hash.

`generate_ms` includes observation preparation and CPU action conversion. On two GPUs it also includes inter-process communication. `prediction_ms` measures the policy call, synchronized with CUDA. Neither includes network transport or preview decoding.

For a controlled comparison, keep the checkpoint, observations, denoising schedule, preview setting, software and GPU type fixed. Change one setting at a time. Report one-GPU and two-GPU results separately. Compare actual actions before claiming equivalence; matching shapes or finite values are insufficient.

`--check-app` runs additional checks after the timed samples: prompt refusal, chunk progression, prompt changes, reset, frame metadata and a decoded preview. It calls the application with the loaded checkpoint without opening a network session. It also repeats a first-chunk request after reset and records the exact match and maximum action difference. These calls are excluded from latency statistics.

To run the opt-in GPU test directly:

```sh
DREAMZERO_TEST_WEIGHTS=/weights DREAMZERO_TEST_CONFIG=config.yml \
  python -m pytest tests/test_gpu.py -q
```

## SDK end-to-end checks

Start the model with `reactor run` on your GPU host, then run from the repository root:

```sh
DREAMZERO_E2E_URL=http://localhost:8080 DREAMZERO_E2E_RESULTS=./e2e-results \
  uv run --python 3.12 --with reactor-sdk==1.13.0 --with numpy --with pytest \
  pytest client/test_e2e.py -v
```

These tests open real stepped and streamed SDK sessions. They check refusal, action metadata, prompt changes, reset within a step, preview pixels and a new session on the resident model. They send synthetic observations and do not measure task success. The JSON reports retain step round-trip and policy timing separately.
