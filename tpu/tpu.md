# Google Cloud TPU (TorchTPU) Support

The native PyTorch `torch_tpu` backend (`device="tpu"`) targets Colab `TPU v6e-1` (32 GiB HBM). The transformer patch has CPU parity tests against the original modeling code, and both strict eager (`default`) and `defer_and_fuse` execution have been validated end-to-end on live Colab `TPU v6e-1` hardware.

## Architecture Summary

1. **Stage 1 — Exact `Qwen3.5-9B` Text Encoding (15.6 GiB BF16)**:
   - Runs unpadded individual prompt forwards on `text_encoder` to preserve Qwen3.5 Gated DeltaNet (GDN) numerical semantics (`pipeline_moss_image.py`).
   - Detaches `[1, L_true, 4096]` hidden states to CPU and stores them in an 8-entry LRU prompt cache, or exports them via `--mode encode-only`.
2. **Stage 2 & 3 — Static-Bucketed `MOSSImageTransformer` (5B DiT) + FP32 `FLUX.2` VAE**:
   - Offloads `text_encoder` before loading `transformer` (~10.2 GiB BF16) and `vae` (~0.3 GiB FP32) onto TPU (`--offload model` / staged), leaving them resident across subsequent seeds and cached prompts.
   - Uses a precomputed float32 frequency buffer for the DiT timestep sinusoid.
   - Uses static bucket padding (`512, 1024, 2048, 4096`), masked SDPA, and cumulative RoPE offsets to keep the DiT's input shapes fixed within a bucket.

## Quick Start (Google Colab `TPU v6e-1`)

```bash
bash tpu/setup-tpu.sh
python3 tpu/run_tpu.py --mode probe
python3 tpu/run_tpu.py --mode generate --prompt "A ceramic teapot on linen"
```

TPU downloads use a RAM-backed cache automatically when tmpfs and host memory have enough space.
Use `MOSS_IMAGE_TPU_DISK_CACHE=1` to use the regular Hugging Face cache on disk.

> Runs strict eager (`default` / `DEFER_NEVER`, `--no-compile`) by default, which is verified end-to-end on `TPU v6e-1`. For ~3x faster warm total generation (`16.72s` vs `49.08s` at `848x1200`, 55 steps; `16.43s` vs `48.61s` denoising-only) and ~8 GiB lower peak HBM (`16.3 GiB` vs `24.5 GiB`), pass `--eager-mode defer_and_fuse`. Pass `--compile` only for experimental DiT compilation debugging (`e0200: RuntimeUnexpectedCoreHalt`).

### Eager / Defer-and-Fuse Benchmarking

```bash
python3 tpu/run_tpu.py \
  --mode benchmark \
  --bench-modes eager,defer_and_fuse \
  --steps 28 \
  --save-report outputs/tpu_benchmark.json
```

Each mode reports its first and subsequent generations separately. The first generation can include prompt encoding, model transfers, and compilation. Cache hits can reduce compilation time on any run. The report includes the actual denoising step count and mean denoising time per step. Use `--prompt-embeds` to compare runs without prompt encoding or stage transfers.

### Decoupled text encoding and generation

Encode on a TPU with enough HBM for the BF16 text encoder and its activations, or pass `--allow-cpu` on a host with sufficient RAM. Copy the resulting file to the generation host. The generation command loads the DiT and VAE and can be evaluated separately on a 16 GiB TPU v5e-1. Use the same model and guidance settings for both commands.

```bash
python3 tpu/run_tpu.py --mode encode-only --prompt "..." --save-embeds outputs/embeds.pt
python3 tpu/run_tpu.py --mode generate --prompt-embeds outputs/embeds.pt --offload none
```

### Persistent Compilation Cache

Set these before running to reuse XLA executables across restarts:

```bash
export TORCH_TPU_TIER2_COMPILATION_CACHE="${HOME}/.cache/torch_tpu/tier2"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${HOME}/.cache/torch_tpu/tier3"
```

## Notes on Pallas SplashAttention

`tpu_runtime.py` lowers attention through `F.scaled_dot_product_attention(..., attn_mask=...)` for exact mathematical parity across padded prompt buckets. Custom Pallas `splash_attention_kernel` bridges via `torch.tpu.pallas.jax_op` (following `google-pytorch/torch_tpu` FLUX optimization recipes) require fixed block-aligned masks (`128`-token alignment) and live TPU verification and are left as an explicit Phase 2 hardware tuning path.

## Live Colab `TPU v6e-1` Validation & Known Errors (`2026-11-13`)

Tested commit `f41afbd` (`feat/torchtpu-support`) on a live Google Colab `V6E1` runtime (`TFRT TPU v6 lite`, PJRT C API `0.116`, `libtpu 0.0.50`, `Python 3.13.18`, `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/content/tpu_cache/tier3`).

### Verified Working Stages

- `bash tpu/setup-tpu.sh`: exited `0`.
- `python3 tpu/run_tpu.py --mode probe`: exited `0`.
- Checkpoint loading (`759/759` weights), Stage 1 `Qwen3.5-9B` prompt encoding (`DEFER_NEVER` + Tier-3 cache), Stage 2 `MOSSImageTransformer` transfer, and (with `--no-compile`) the multi-step DiT denoising loop + `FlowMatchEulerDiscreteScheduler.step()` completed on hardware.

### Error 1 — `--compile` Triggers Asynchronous TPU Core Halt (`e0200` / `vmem_address_out_of_range`)

**Commands**

```bash
python3 tpu/run_tpu.py --mode generate --offload model --compile --steps 28 --seed 42 --output outputs/colab_tpu_sample.png
python3 tpu/run_tpu.py --mode generate --offload model --steps 2 --seed 42 --output outputs/eager_test.png  # tested on f41afbd when --compile defaulted to True
```

**Observed Error**

Surfaced at step 0 during the first host D2H sync inside `diffusers/schedulers/scheduling_flow_match_euler_discrete.py:414` (`(schedule_timesteps == timestep).nonzero()`):

```text
E0000 ... pjrt_transfer.cc:391] Async D2H ToLiteral transfer failed: INTERNAL: E0200: RuntimeUnexpectedCoreHalt: Program or fatal error occurred; computation may be invalid: (launch_id=0)
...
RuntimeError: failed to copy buffer from device into host buffer -- the XLA compiler failed with: e0200: RuntimeUnexpectedCoreHalt: Program or fatal error occurred; computation may be invalid: (launch_id=0)Assertion args: INTERNAL: Fatal error occurred during TPU execution. ...
Detailed error: Node 1 halted unexpectedly at tag:pc SparseCoreSequencer:0:0x5 (from SparseCoreSequencer:0:0x5): no debugging message found for this tag:pc. ; HLO module: tt_jit_nonzero
  0x0x0_SC1 SparseCoreTileExecuteCoreSequencer:0..15 halted at tag:pc SparseCoreTileExecuteCoreSequencer:0:0 ...
Summary of reported fatal error messages by core:
core location: 0x0x0_SC1, error type: UserFatal, message: *** [Physical location: tpu15:pe4:2] A UserFatal error has occurred, all pending programs will fail, and results may be corrupted.
...
Anomalies (4 total):
[user] chip_tc_tcs_tc_main_fatal_errors_tc_vector_program_errors0.node: 0x20000000000: vmem_address_out_of_range_vld0
[user] chip_tc_tcs_tc_main_fatal_errors_tc_vector_program_errors0.node: 0x40000000000: vmem_address_out_of_range_vld1
```

**Isolation Results**

- Running `torch.tensor([False, True, False], device="tpu").nonzero()` in isolation exited `0` (`tensor([[1]], device='tpu:0')`).
- Running the scheduler shape `t = torch.linspace(1000.0, 1.0, 29, device="tpu"); (t[:-1] == t[0]).nonzero()` under `torch.tpu.EagerMode.DEFER_NEVER` with `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` set also exited `0` (`tensor([[0]], device='tpu:0')`).
- Passing `--no-compile` bypassed the core halt and finished the DiT denoising loop and `scheduler.step()` cleanly.
- **Scope proven by logs**: Enabling `--compile` (`transformer.compile(backend="tpu", dynamic=False, fullgraph=True)`) causes `e0200: RuntimeUnexpectedCoreHalt` (`vmem_address_out_of_range_vld0`/`vld1`) surfaced at the first synchronous host D2H copy (`scheduler.step()` -> `.nonzero()`), whereas `--no-compile` completes Qwen3.5 encoding and the DiT + scheduler denoising loop cleanly. The exact compiled operation or lowering triggering the VMEM out-of-range fault has not yet been isolated with intermediate device synchronizations.

### Error 2 — Eager `--no-compile` Fails in VAE Decode (`torch.group_norm` Channels-Last View)

**Command**

```bash
python3 tpu/run_tpu.py --mode generate --offload model --no-compile --steps 2 --seed 42 --output outputs/nocompile_test.png
```

**Observed Error**

After `Qwen3.5-9B` encoding, 2 DiT denoising steps, and `FlowMatchEulerDiscreteScheduler` all completed, Stage 3 (`AutoencoderKLFlux2` decode in `vae.py:300` -> `unet_2d_blocks.py:746` -> `resnet.py:326` -> `F.group_norm`) failed:

```text
RuntimeError: view(): cannot create a view of shape [1, 32, 196608] from the input tensor of shape [1, 384, 128, 128] and strides [384, 1, 49152, 384]; consider creating a new tensor using reshape() instead of taking a view
```

- **Root cause**: `torch_tpu` convolution/decoder blocks output a non-contiguous channels-last (`NHWC`, strides `[384, 1, 49152, 384]`) tensor, and eager `torch.group_norm` attempts a `.view(...)` reshape without calling `.contiguous()`.

### Error 3 — Broadcast 4D SDPA Masks (`[1, 1, 1, S]`) Produce `NaN`s at `S >= 512` (All-Black Image)

Tested commit `70a8c7e` on `TPU v6e-1` with `--mode generate --offload model --height 1200 --width 848 --steps 55`:

- **Observed Symptom**: All three stages (`Qwen3.5-9B`, 55-step DiT denoising, `AutoencoderKLFlux2` decode with the `GroupNorm` hook) exited `0` (`peak_hbm_gib: 24.153`), but saved a solid-black `(1200, 848, 3)` PNG (`min=0, max=0`) alongside `RuntimeWarning: invalid value encountered in cast` in `diffusers/image_processor.py`.
- **Bisection & Root Cause**:
  - `Qwen3.5-9B` embeddings were clean (`nan=False`, `absmax=50.0`), while DiT output latents already contained `NaN`.
  - Forward hooks isolated the first `NaN` immediately inside attention output projections: `text_embedder.blocks.0.proj` when bucket-padded (`S=512`), and `layers.0.proj.o` when unpadded (`S=3975+30=4005`, after `text_embedder` passed at `S=30`).
  - Standalone TPU `F.scaled_dot_product_attention` tests showed `torch_tpu`'s lowered Mosaic/cuDNN kernel silently miscomputes 4D boolean masks that broadcast the query axis (`[1, 1, 1, S]`, `maxdiff=2.1367` at `S=30`) and emits `NaN` for `S >= 512` (`S=512`, `S=4005`). Expanding the mask to `[1, 1, Q, S]` matches CPU `float32` SDPA to BF16 precision even with poisoned masked keys/values (`valid_maxdiff=0.0029` at `S=4487`, `2.38 ms`).

### Colab CLI Operational Note (`google-colab-cli 0.8.0`)

- `colab new --tpu v6e1` may hit `503 Service Unavailable`, proxy idle disconnects (`RemoteDisconnected`), or `ReadTimeout` (`COLAB_CLI_REQUEST_TIMEOUT=70`) if backend allocation finishes after the HTTP wait window. Always run `colab sessions` afterward to inspect or adopt `[?]` background-allocated runtimes and `colab stop -s <name>` when finished so compute units are not leaked.

### Applied Fixes & End-to-End Hardware Verification (`TPU v6e-1`)

1. **Default `--compile` to `False` (`tpu/run_tpu.py`)**: `tpu/run_tpu.py` defaults `--compile` to `False` and `--bench-modes` to `"eager,defer_and_fuse"`, keeping `--compile` opt-in while the compiled DiT `vmem_address_out_of_range` lowering issue is isolated further.
2. **Enforce NCHW-contiguous `GroupNorm` inputs (`tpu/tpu_runtime.py`)**: `ensure_group_norm_contiguous(vae)` installs an idempotent forward pre-hook on all `nn.GroupNorm` submodules in `StagedTPUImagePipeline` so channels-last convolution outputs (`[384, 1, 49152, 384]`) are converted via `.contiguous()` before `torch.group_norm` calls `.view(...)`.
3. **Expand 4D static SDPA mask query dimension (`tpu/tpu_runtime.py`)**: `MOSSImageTransformer.forward` expands `text_attn_mask` and `joint_attn_mask` from `[1, 1, 1, S]` to `[1, 1, Q, S]`, backed by a guard in `_attention`, eliminating `NaN`s in both padded text projection and joint DiT blocks.

Verified on live Colab `TPU v6e-1` (`TFRT TPU v6 lite`, `PJRT 0.116`, `libtpu 0.0.50`, `OpenMOSS-Team/MOSS-Image-Base`, `848x1200`, `55 steps`, `guidance_scale=3.0`, `uncond_skip_layers=(22,)`, `329` prompt tokens -> bucket `512`, `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/content/tpu_cache/tier3`):

- **Stage 1 standalone `Qwen3.5-9B` encode (`--mode encode-only`)**: `39.52s`, `15.801 GiB` peak HBM.
- **Full cold end-to-end strict eager run (`prompt` + `negative_prompt` -> PNG)**: `124.19s` total (`52.978s` denoising / `963.23 ms/step`, `24.496 GiB` peak HBM, verified valid RGB image).
- **`--mode benchmark` (`848x1200`, 55 steps, cached `--prompt-embeds`)**:
  - `eager` (`DEFER_NEVER`): Run 1 cold total `70.201s` (`48.883s` denoising, `24.472 GiB`); Run 2 warm total `49.082s` (`48.612s` denoising / `883.85 ms/step`, `24.473 GiB`).
  - `defer_and_fuse`: Run 1 cold total `35.556s` (`16.605s` denoising, `14.963 GiB`); Run 2 warm total `16.724s` (`16.434s` denoising / `298.79 ms/step`, `16.331 GiB`).
- **10-seed end-to-end batch (`StagedTPUImagePipeline`, `defer_and_fuse`, `--offload model`)**: `246.17s` total across all 10 distinct random seeds (`load_seconds=3.326s`, Run 1 cold `95.563s` including Stage 1 Qwen encoding + compile, Runs 2–10 warm mean `16.734s` total / `16.429s` denoising / `298.71 ms/step`, `16.248 GiB` peak HBM; 10/10 verified valid RGB images).

## Stage 1 (`Qwen3.5-9B`) Profiling, Prompt Cache & Experimental Pallas GDN (`tpu_qwen35.py`)

### Stage-1 Breakdown Benchmarking

To separate tokenizer/checkpoint loading, CPU-to-TPU weight transfer (`h2d_transfer_seconds`), cold compilation (`cold_encode_seconds`), negative prompt encoding, warm steady-state encoding (`warm_encode_seconds`), and D2H copy (`cold_d2h_seconds`):

```bash
python3 tpu/run_tpu.py \
  --mode encode-bench \
  --bench-modes eager,defer_and_fuse \
  --runs 3 \
  --save-report outputs/encode_bench.json
```

### Persistent Disk Prompt Cache

For repeated CLI invocations sharing prompts (such as the default `negative_prompt=""`), pass `--prompt-cache-dir` to reuse persisted CPU text embeddings across process restarts:

```bash
python3 tpu/run_tpu.py \
  --mode generate \
  --eager-mode defer_and_fuse \
  --prompt-cache-dir outputs/prompt_cache \
  --prompt "A ceramic teapot on linen"
```

### Experimental Pallas Gated DeltaNet (`deltanet-tpu`)

Qwen3.5-9B uses 24 `GatedDeltaNet` linear-attention layers (`Hk=16, Hv=32, D=128`, `use_cache=False` during encoding). An opt-in `torch.tpu.pallas.jax_op` bridge (`tpu_qwen35.py`) can route `torch_chunk_gated_delta_rule` through `deltanet_tpu.chunk_gdn` with `128`-aligned static causal buckets (`128, 256, 384, 512, 768, 1024`).

> [!WARNING]
> `deltanet-tpu` requires Python `>= 3.12` and a compatible `jax` runtime (preinstalled on Colab TPU). On TorchTPU runtimes, install it with `--no-deps` so it does not overwrite TorchTPU's pinned `libtpu`:
> `pip install --no-deps deltanet-tpu`

Run standalone single-layer & recurrence A/B verification (no full 18 GB checkpoint download required) or end-to-end fast split-stage generation (Stage 1 `Qwen3.5-9B` automatically runs in `default` / `DEFER_NEVER` + Pallas when `--text-eager-mode auto` is active alongside `--gdn-pallas`, while Stage 2/3 DiT+VAE runs in `defer_and_fuse`):

```bash
python3 tpu/bench_qwen35_gdn_tpu.py --include-layer --save-report outputs/gdn_ab.json
python3 tpu/run_tpu.py --mode encode-bench --bench-modes eager,defer_and_fuse,eager+gdn_pallas,defer_and_fuse+gdn_pallas --save-report outputs/encode_bench_gdn.json
python3 tpu/run_tpu.py --mode generate --eager-mode defer_and_fuse --gdn-pallas --prompt "A ceramic teapot on linen"
```

Pass `--text-eager-mode default` or `--text-eager-mode defer_and_fuse` to override Stage 1 explicitly (`--text-eager-mode auto` resolves to `default` when `--gdn-pallas` is enabled and `--eager-mode` otherwise).

## Stage Switching Optimization (`drop_tpu` & Async `non_blocking` H2D)

When switching between Stage 1 (`Qwen3.5-9B`, `15.63 GiB`) and Stage 2/3 (`transformer` `9.4 GiB` + `vae`), classic `d2h` offloading copies TPU weights back to CPU RAM on every switch (`~16–33s` per offload) and then restores them synchronously over H2D.

Plan A adds three optimizations in [tpu/tpu_runtime.py](tpu_runtime.py) and `--mode stage-switch-bench` in [tpu/run_tpu.py](run_tpu.py):

1. **`drop_tpu` zero-D2H stage eviction (`--stage-switch-mode drop_tpu`)**: keeps an immutable CPU snapshot of each staged module in host RAM, releases off-stage TPU parameters/buffers directly to zero-byte `meta` tensors (`d2h = 0.0s`, `drop_tpu ≈ 0.011–0.021s`) without device-to-host copies, and restores directly from the CPU snapshot upon re-entry.
2. **Async `non_blocking=True` H2D transfers**: `_restore_module_from_cpu_snapshot` and `_activate_stage` batch all tensor `.to(tpu, non_blocking=True)` transfers under a single `torch.tpu.synchronize()`, raising H2D throughput from `~0.53 GiB/s` (synchronous) to `~1.96 GiB/s` on a synthetic 400-tensor `12.50 GiB` benchmark (`23.777s` -> `6.383s`; pre-pinned + `non_blocking=True` reached `6.101s` / `2.05 GiB/s` at a `0.446s` pinning cost).
3. **Resident `vae` across `diffusion <-> text` prompt switches**: once `vae` (`~0.3 GiB`) is loaded onto TPU for diffusion, it remains resident across subsequent prompt switches instead of offloading and reloading (cold `cpu -> text` calls such as `--mode encode-only` still avoid loading `vae`, and explicit `_activate_stage("cpu")` still evicts it). *(The `ec71fec` benchmark table below was collected prior to this `vae`-residency follow-up.)*

### Validated TPU v6e-1 Stage-Switch Results (`848x1200`, `defer_and_fuse + --gdn-pallas`, `seed=136879772936584`)

Run the end-to-end A/B stage-switch benchmark (`d2h` vs `drop_tpu`, with an interactive `diffusion -> text -> diffusion` switch between Run 1 and Run 2):

```bash
python3 tpu/run_tpu.py --mode stage-switch-bench --switch-modes d2h,drop_tpu --eager-mode defer_and_fuse --gdn-pallas --height 848 --width 1200 --steps 55 --seed 136879772936584 --save-report outputs/stage_switch_55.json
```

| Configuration | Steps | Switch Mode | Total Switch (s) | D2H (s) | Drop + GC + Cache (s) | H2D (s) | Denoise (s) | Warm Run 2 Total (s) | Peak HBM (GiB) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pre-async H2D baseline (`9fb8310`) | 4 | `d2h` (Run 2 interactive) | 100.734 | 50.250 | 0.000 | 50.484 | 1.224 (`305.91 ms/step`) | 103.286 | — |
| Pre-async H2D + `drop_tpu` (`9fb8310`) | 4 | `drop_tpu` (Run 2 interactive) | 48.965 | 0.000 | 0.938 | 48.025 | 1.223 (`305.83 ms/step`) | 50.624 | — |
| Async `non_blocking` H2D (`ec71fec`) | 55 | `d2h` (Run 1) | 48.996 | 29.666 | 0.000 | 19.330 | 20.004 (`363.7 ms/step`) | 70.960 *(Run 1)* | 15.662 |
| Async `non_blocking` H2D (`ec71fec`) | 55 | `d2h` (Run 2 interactive) | 67.077 | 49.444 | 0.000 | 17.633 | 16.551 (`300.93 ms/step`) | 84.412 | 20.665 |
| Async `non_blocking` H2D (`ec71fec`) | 55 | `drop_tpu` (Run 1) | 18.678 | 0.000 | 0.491 | 18.186 | 16.543 (`300.77 ms/step`) | 35.720 *(Run 1)* | 20.665 |
| Async `non_blocking` H2D (`ec71fec`) | 55 | `drop_tpu` (Run 2 interactive) | **20.745** | **0.000** | **0.995** | **19.750** | **16.543** (`300.78 ms/step`) | **37.711** | **20.667** |

Key hardware observations from the `55-step` run (`host RAM: 172.948 GiB total`, `168.553 GiB available` before pipeline load):

- **Interactive switch & end-to-end speedup**: On `ec71fec` (55 steps), `drop_tpu` cuts Run 2 interactive stage-switch overhead from `67.0769s` to `20.7452s` (**3.233x** faster, saving `46.3317s`) and Run 2 end-to-end time from `84.412s` to `37.711s` (saving `46.701s`). Across both steps of Plan A (`9fb8310` pre-async `d2h` `100.7338s` -> `ec71fec` async `drop_tpu` `20.7452s`), interactive switch latency dropped by **4.86x**. Run 1 switch overhead also fell from `48.9963s` to `18.6778s` (saving `30.3185s`).
- **Bitwise image parity**: Both Run 1 and Run 2 `drop_tpu` outputs match `d2h` baselines with `cosine_similarity = 1.0`, `mean_abs_diff = 0.0`, `max_abs_diff = 0.0`, and `psnr_db = Infinity` (`torch.float64` parity accumulation).
- **Host RAM footprint**: After `drop_tpu` Run 2 on the `172.948 GiB` Colab TPU v6e-1 VM, process RSS was `30.910 GiB` with `138.244 GiB` host RAM remaining free (vs `23.652 GiB` RSS / `145.518 GiB` free under `d2h`), comfortably fitting the CPU snapshot cache.
- **Residual `~5.0 GiB` HBM plateau**: After the first full pipeline pass, both `d2h` (`5.002–5.025 GiB`) and `drop_tpu` (`5.008–5.025 GiB`) retain a `~5.0 GiB` floor in `torch.tpu.memory_allocated()`, even after deleting the pipeline and running `gc.collect()` + `empty_cache()`, while Python `gc.get_objects()` reports `0` live TPU tensors (`0.0000 GiB`, or at most `1–3` tensors / `0.0010 GiB`). This is a one-time TorchTPU/PJRT C++ runtime/executable buffer plateau rather than a Python weight leak; it stays flat across repeated switches and keeps peak HBM at `20.667 GiB / 31.24 GiB`. Note: external callers computing forward activations with `requires_grad=True` retained across switches can pin weight references in autograd, so inference runs cleanly under `torch.inference_mode()` / `torch.no_grad()`.

