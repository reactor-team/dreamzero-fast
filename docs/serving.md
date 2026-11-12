# Serving DreamZero

Run the commands in the [README](../README.md) from `dreamzero/`. [`reactor.yaml`](../dreamzero/reactor.yaml) selects `dreamzero:DreamZero`, `config.yml` and the weights directory.

## Weights

The weights directory must contain:

- `checkpoint/`: the DreamZero-DROID configuration, checkpoint shards and supporting experiment metadata.
- `hf/`: a Hugging Face cache containing Wan2.1-I2V-14B-480P's denoiser shards, T5, CLIP and VAE weights, plus the `google/umt5-xxl` tokenizer.
- `checkpoint/tensorrt/wan/WanModel_nvfp4.trt`: required when TensorRT is enabled.

The model sets `HF_HOME` to this cache and loads offline. Download the checkpoint from [GEAR-Dreams/DreamZero-DROID](https://huggingface.co/GEAR-Dreams/DreamZero-DROID). TensorRT plans are platform-specific; build the engine on the target host from the checkpoint's ONNX graph and external data:

```sh
python scripts/build_trt_engine_local.py --weights /path/to/weights
```

## Inputs and episode state

Stream `exterior_1`, `exterior_2` and `wrist` as RGB camera tracks. Set `prompt`, `joint_position` (seven radians) and `gripper_position` (0 open, 1 closed) with `set_prompt`, `set_joint_position` and `set_gripper_position`, or in a stepped request's `state`.

The application waits for a prompt and at least one fresh frame from every camera. Frames received during a refusal are retained until cameras share a `capture_time_us` (within 1 ms). Each accepted step passes at most four frames per camera ending at that shared moment. Unmatched newer frames stay pending so a lagging track can catch up. Unstamped frames or skew beyond 1 ms refuse the step.

The model retains four frames per camera. The first chunk uses only the newest frame in each view. Later chunks use the four-frame window, repeating the oldest frame while the window fills. A changed prompt clears the old episode and starts chunk zero.

Measured joint state lets the policy return absolute joint targets. Without it, the model uses zeros and the outputs remain relative deltas. The robot application must distinguish those cases and enforce execution timing and limits.

## Outputs and reset

The application sends `episode_started` for chunk zero and `action_chunk` after every prediction. Each action chunk contains its index, the action array and `inference_seconds`. That timer covers the policy call and conversion to CPU actions; it excludes camera preparation, inter-process communication, preview decoding and network transport.

`set_video_preview` enables decoded frames on `predicted_video`. With previews off, the track carries one black frame per step. Each frame also carries the JSON `action_chunk` wire envelope plus the same action fields at the top level in metadata, so a stepped client can read actions from its returned media. Streaming clients receive the message directly.

`reset` clears model history and pending application frames, but keeps the prompt and robot state. Reset also flushes queued preview media; ending a session clears the model episode and flushes output too. Inference errors propagate and end the session rather than silently continuing with a partial cache.

## Stepped sessions

A stepped request supplies camera frames and optional state updates together. The response contains the action chunk in the `predicted_video` frame metadata. See the [client example](../client/policy_client.py) for request construction and action parsing.

## Streamed sessions

Publish all three camera tracks continuously and receive `action_chunk` messages independently. Use the chunk index to track predictions within an episode. A closed camera track refuses further steps; reconnect the session to restore its inputs.
