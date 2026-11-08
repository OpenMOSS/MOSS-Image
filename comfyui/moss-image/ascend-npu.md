# Running MOSS-Image on Ascend NPU (910B2C)

This guide covers both official inference paths — the Diffusers native
pipeline and the ComfyUI native plugin — on Huawei Ascend NPUs via the
open-source [device-adapter](https://gitcode.com/Ascend-SACT/device-adapter).
Model code is used **as-is**; all changes live in two small adaptation layers.

## 1. Dependencies

| Item | Version tested | Requirement |
|------|----------------|-------------|
| Python | 3.13 | >= 3.10 |
| torch / torch_npu | 2.11.0+cpu | CPU-only torch is sufficient |
| diffusers | 0.39.0 | >= 0.38 |
| transformers | 5.12.1 | >= 5.3 (needs `Qwen3_5Model`) |
| accelerate / safetensors / Pillow | 1.12.0 / 0.8.0 / 11.3.0 | satisfied |
| device-adapter | 0.2.0 | from gitcode |

## 2. Track A — Diffusers native pipeline

```python
import device_adapter
# 适配到加速设备
device_adapter.adapt_to_accelerator_device()

import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "/path/to/MOSS-Image-SFT",           # or "OpenMOSS-Team/MOSS-Image-SFT"
    custom_pipeline="pipeline_moss_image",
    trust_remote_code=True,
    torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
    local_files_only=True,
)
pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload(device="npu")

image = pipe(
    prompt,
    height=1024, width=1024,
    num_inference_steps=50,
    guidance_scale=3.0,
    uncond_skip_layers=[22],
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("out.png")
```

Notes:

- `torch_dtype` vs `dtype`: newer diffusers renamed the kwarg; the fallback
  `except TypeError` is shown in the packaged script.
- `torch.Generator("cpu")` is unaffected by the NPU mapping — diffusers
  `randn_tensor` supports CPU generators on an NPU device; initial noise is
  sampled in fp32 on CPU, matching the README.
- Measured on a single 910B2C (65 GB HBM): 1024×1024 / 50 steps / CFG 3.0 →
  **~42–45 s warm**, peak memory **15.67 GiB**.

### Direct load (`pipe.to("npu")`)

If ≥ 32 GB HBM is free, load everything resident:

```python
pipe.to("npu")
```

| Model | Warm latency (median of 4) | Peak memory |
|-------|---------------------------|-------------|
| Base  | 34.0s | 28.93 GiB |
| SFT   | 34.1s | 28.93 GiB |

~24% faster than offload mode. Keep `enable_model_cpu_offload(device="npu")`
as the default for tight-memory deployments; switch to direct load when
the card is idle.

## 3. Track B — ComfyUI native plugin

### 3.1 Launch ComfyUI on NPU with zero code changes

`sitecustomize.py` is auto-imported by Python at startup; putting the
device-adapter on `PYTHONPATH` injects the NPU mapping into every ComfyUI
process before any torch import:

```bash
cd comfyui_track
PYTHONPATH=$PWD/npulaunch setsid bash -c \
  'exec python3 -u ComfyUI/main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch' \
  > /tmp/comfy_server.log 2>&1 < /dev/null &
```

The server log should show `Device: npu`.

### 3.2 One compatibility patch (4 lines)

`transfer_to_npu` patches `nn.Module.to(device)` to call
`self.cast_weight(device)` — passing a `torch.device`. ComfyUI's custom ops
expect `cast_weight(input_tensor)` and immediately access `input.dtype`,
raising:

```
AttributeError: 'torch.device' object has no attribute 'dtype'
```

Guard at the single choke point in `comfy/ops.py`:

```python
def cast_bias_weight(s, input=None, dtype=None, device=None, ...):
    if isinstance(input, torch.device):
        if device is None:
            device = input
        input = None
    ...
```

All custom-op `cast_weight` implementations funnel through this function,
so one guard fixes Linear / LayerNorm / RMSNorm / Conv / etc. The CUDA path is
unchanged.

### 3.3 Generate via the API

Load `example_workflows/moss-image-api.json`, point the
`MOSSImageHFLoader` node's `model_path` at the local model directory, and POST
to `/prompt`. Workflow defaults: 50 steps, CFG 3, Shift 5, SLG layer 22,
1024×1024.

Measured warm latency: **36.0 s** per image (3 consecutive runs, both Base
and SFT). ComfyUI keeps weights resident on the NPU, which is why Track B is
~6–9 s faster than the offloading Diffusers setup.

## 4. Operator compatibility notes (910B2C, CANN 9.0.0)

| Concern | Status |
|---------|--------|
| bf16 `scaled_dot_product_attention` | OK, SDPA path untouched |
| fp64 time embeddings (`times.double()`, fp64 cos/sin) | OK on 910B |
| `searchsorted` / `unique_consecutive` (scheduler) | OK (runs on CPU) |
| Qwen3.5 GDN text encoder (transformers 5.x pure-torch) | OK |
| AutoencoderKLFlux2 fp32 decode | OK |
