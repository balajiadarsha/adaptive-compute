# Known issue: GPTQ quantization (gptqmodel) fails on Intel XPU (Aurora)

## Summary

`quantize_model.py` (GPTQ int8/int4/int3 via `gptqmodel`) does not currently
work on Aurora's Intel XPU nodes. It fails partway through layer 0's forward
pass with:

```
RuntimeError: could not create a memory
```

This is not a bug in this repo's scripts — it's an upstream compatibility gap
between `gptqmodel`'s multithreaded execution engine and Intel's XPU PyTorch
backend. `gptqmodel` appears to be built and tested primarily against CUDA.

## Environment

- Node: Aurora compute node (e.g. `x4217c0s5b0n0`), 12 XPU devices/node
- `module load frameworks` (2025.3.1) + `.safety_env` venv
  (`--system-site-packages`, layered on top of `frameworks`)
- `torch 2.10.0a0+git449b176`, `transformers 5.14.1`, `gptqmodel 7.3.2`
- Model: `l3lab/L1-Qwen-1.5B-Exact` (Qwen2 architecture, 1.5B params)

## How to reproduce

```bash
module load frameworks
source /home/abalaji/projects/agpt/safety-eval/.safety_env/bin/activate
export HF_HOME=/flare/ModCon/abalaji/cache/model_weights
export HF_HUB_CACHE=/flare/ModCon/abalaji/cache/model_weights/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ZE_AFFINITY_MASK=0   # restrict to 1 XPU device, see below

cd adaptive-compute/quantize
python quantize_model.py \
  --model-id <local-cache-snapshot-path> \
  --bits 8 --calib-dataset safechain --calib-samples 32 \
  --output-dir /tmp/test-int8
```

## Debugging trail

1. **Zero XPU devices visible.** Running from a login/UAN node
   (`aurora-uan-*`) or a shell without `module load frameworks`, `torch.xpu
   .is_available()` is `False`. `gptqmodel`'s device coercion
   (`threadx.py::_coerce_device`) silently falls back to CPU in this case.
   The CPU path then crashes in a background warmup thread doing a tiny
   `cholesky`/`eigh` sanity check (`linalg_warmup.py::_run_cholesky_and_eigh`)
   with the same `RuntimeError: could not create a memory`. This is a red
   herring for the real issue below — always confirm `torch.xpu
   .is_available()` is `True` (12 devices) before investigating further.
   `module load frameworks` must run in the same shell/job step as the
   Python invocation.

2. **Compute nodes have no outbound internet access**, proxy included
   (`proxy.alcf.anl.gov:3128` CONNECT tunnels time out for every HTTPS host
   tested, not just huggingface.co). `gptqmodel.models.auto.GPTQModel.load()`
   unconditionally calls `list_repo_files()` (a live Hub API call) even when
   `HF_HUB_OFFLINE=1` is set — `transformers`/`datasets` respect the offline
   flag, `gptqmodel`'s own Hub call does not. **Workaround:** pass the
   resolved local snapshot directory as `--model-id` (e.g.
   `$HF_HUB_CACHE/models--<org>--<name>/snapshots/<hash>`) instead of the
   `org/name` HF identifier, so `gptqmodel` treats it as a local path and
   skips the Hub API call.

3. **Multi-device (12 XPU) run**: with the local-path workaround, the model
   loads correctly onto all 12 XPUs and quantization starts (`Quantizing
   layer 0 of 27`). It crashes during `Forward: replicate to 12 devices`,
   immediately preceded by:
   ```
   INFO  Clone: fast clone failed module 'torch._C' has no attribute '_broadcast_coalesced'
   ```
   `torch._C._broadcast_coalesced` is a CUDA-only PyTorch internal that
   `gptqmodel`'s multi-device tensor replication path assumes exists. It logs
   the failure and falls back to a slower clone path, but the subsequent
   per-device warmup thread then fails with `RuntimeError: could not create a
   memory` (same error as case 1, but now genuinely on XPU, one thread per
   device, 12 threads total).

4. **Restricting to 1 XPU device** (`ZE_AFFINITY_MASK=0`, confirmed via
   `torch.xpu.device_count() == 1`) avoids the multi-device replication path
   entirely. Quantization gets further — past the warmup, into the actual
   forward pass — before crashing in `F.linear` inside
   `gptqmodel/nn_modules/hooked_linear.py::forward` on
   `model.layers.0.self_attn.q_proj`, again with `RuntimeError: could not
   create a memory`.

5. **Isolated repro attempts did NOT reproduce the crash**, ruling out the
   simplest explanations:
   - Plain `F.linear` on small tensors on XPU: **works**.
   - Plain `F.linear` at realistic model dimensions (3768×1536 @ 1536×1536,
     fp16, matching the actual calibration batch/hidden size): **works**,
     with ~68GB free XPU memory reported by `torch.xpu.mem_get_info()`.
   - Same `F.linear` call issued from a plain Python background thread
     (`threading.Thread`), mimicking `gptqmodel`'s `threadx.py` worker-thread
     pattern: **works**.

   So the failure is not simple tensor size, device memory pressure, or
   "running on a non-main thread" in isolation — it's something specific to
   `gptqmodel`'s actual execution engine (`threadx.py`'s thread pool +
   `hooked_linear.py`'s forward wrapping + whatever device/stream/context
   state it manages per-thread), which has not been reproduced standalone.

## Hypothesis (unconfirmed)

`gptqmodel`'s threaded dispatch (`utils/threadx.py`) manages per-thread
SYCL/Level-Zero device context in a way that assumes CUDA context semantics
(implicit context-per-thread via the CUDA driver). Intel's XPU backend may
require explicit per-thread context/queue setup that `gptqmodel` doesn't
perform, causing oneDNN/SYCL to fail allocating a memory object under
`gptqmodel`'s specific thread-pool + hook-wrapping combination — even though
the same op succeeds with a hand-rolled single background thread.

## Next steps (not yet attempted)

- Check `gptqmodel`'s GitHub issues/changelog for existing XPU reports before
  further manual repro.
- Try setting `GPTQMODEL_*`-style env vars or a `threads_per_device`-style
  constructor argument, if one exists, to force single-threaded (serial)
  execution and avoid `threadx.py`'s pool entirely.
- Consider `quantize_model_activation.py` (llmcompressor, not gptqmodel) as
  an alternative path for weight-only quantization if gptqmodel's XPU support
  remains blocked — llmcompressor's execution model may not share the same
  threading assumptions (untested on XPU as of this writing).
- File an upstream issue with ModelCloud/gptqmodel if no existing report
  covers this.
