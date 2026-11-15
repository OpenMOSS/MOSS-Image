<div align="center">
  <p>
    <img src="../assets/lockup-light.svg#gh-light-mode-only" alt="MOSS-Image" width="460">
    <img src="../assets/lockup-dark.svg#gh-dark-mode-only" alt="MOSS-Image" width="460">
  </p>
  <p>
    <a href="moss_image_torchtpu_colab.ipynb"><img alt="Colab TPU" src="https://img.shields.io/badge/Colab-TPU%20v6e--1-F9AB00?style=flat-square&amp;logo=googlecolab&amp;logoColor=white"></a>
    &nbsp;
    <a href="tpu.md"><img alt="TorchTPU Docs" src="https://img.shields.io/badge/Docs-TorchTPU-4285F4?style=flat-square&amp;logo=googlecloud&amp;logoColor=white"></a>
    &nbsp;
    <a href="../README.md"><img alt="MOSS-Image Home" src="https://img.shields.io/badge/MOSS--Image-Main%20README-0f766e?style=flat-square"></a>
  </p>
  <p>
    <a href="README.md">简体中文</a> &nbsp;|&nbsp; <strong>English</strong> &nbsp;|&nbsp; <a href="README_JA.md">日本語</a> &nbsp;|&nbsp; <a href="../README.md">Main README</a>
  </p>
</div>

# MOSS-Image on Google Cloud TPU (`torch_tpu`)

Run **[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** natively in PyTorch on **Google Cloud TPU (`TPU v6e-1`, 31.24 GiB)** with BF16 inference, FP32 VAE decoding, and image quality matching CUDA BF16.

> [!TIP]
> **Memory Recommendation**: If your device has **less than 40 GB** of HBM / VRAM (e.g., `TPU v6e-1` with `31.24 GiB`), enable **`offload`** when running **BF16** weights (`--offload auto`, which enables offloading automatically below `40 GiB`), since text + diffusion weights cannot stay resident together in `31.24 GiB`.

---

## 1. Visual Quality (CUDA Reference vs. TPU)

Same prompt and seed (`848×1200`, 55 steps, `seed = 136879772936584`):

| Original CUDA BF16 Reference | TPU `v6e-1` BF16 Generated |
| :---: | :---: |
| <img src="assets/cuda_bf16_ref.png" alt="Original CUDA BF16 Reference" width="360"> | <img src="assets/tpu_v6e_pallas.png" alt="TPU v6e-1 BF16 Generated Image" width="360"> |

- **Pixel parity**: Cosine similarity **0.9980** vs. CUDA BF16, mean pixel difference **6.15 / 255** (`PSNR 26.80 dB`); `drop_tpu` offload matches `d2h` exactly (cosine **1.0**, max diff **0**).

---

## 2. Speed, Operator Efficiency & Offload Performance

Measured on live Colab **`TPU v6e-1`** (`31.24 GiB` HBM, `848×1200`, 55 steps, CFG `3.0`).

### Generation & VAE Decode Speeds

| Scenario | Total Time | Time Breakdown | Peak HBM |
| :--- | :---: | :--- | :---: |
| **Warm Repeat (Same Prompt)** | **16.72s** | Denoising 16.43s + VAE decode & postprocess ~0.29s | 16.33 GiB |
| **10-Seed Mean (Runs 2–10)** | **16.73s** | ~299 ms/step (246.17s total incl. cold Run 1 at 95.56s) | 16.25 GiB |
| **Changing Prompts (`drop_tpu` Offload)** | **37.71s** | Stage switch 20.75s + denoising 16.54s + encode/VAE ~0.42s | 20.67 GiB |
| **Fresh CLI (Disk Prompt Cache Hit)** | **41.62s** | Model load 2.20s + denoising 16.60s + transfer/VAE | 15.24 GiB |
| **Cold First Run (Pre-Plan A Baseline)** | 88.42s | Without Pallas 115.17s; without operator fusion 124.19s | 15.79 GiB |

> Additional baselines: warm repeat without fusion is **49.08s** (denoising 48.61s); cold diffusion + VAE alone is **35.56s** (`70.20s` unfused); changing prompts with async `d2h` is **84.41s**.

### Theoretical Operator Efficiency (`TPU v6e-1` Peak: `918 TFLOPs` BF16)

- **Diffusion Backbone (`defer_and_fuse`)**: **352.8 TFLOPs** achieved (**38.4% MFU**), **2.96x** faster than unfused eager (`119.3 TFLOPs` / `13.0% MFU`).
- **Text Encoder (`Pallas GDN`)**: **2.03 ms** per layer (**3.08x** speedup) and **1.08 ms** recurrence (**4.57x** speedup); forward pass takes **0.31s** cold, **0.16s** distinct warm, and **0.09s** repeat (vs. 1.99s cold / 1.17s warm without Pallas).

### Offload Stage-Switching Speeds (`< 40 GB` Memory)

When prompts change under `--offload model`, `Qwen3.5-9B` (`15.63 GiB`) swaps with DiT + VAE (`9.70 GiB`):

| Offload Mode | Prompt-Switch Time | End-to-End Time | Speedup |
| :--- | :---: | :---: | :---: |
| Sync transfer + `d2h` *(4-step baseline)* | 100.73s | 103.29s *(4 steps)* | 1.00x |
| Async H2D + `d2h` *(55 steps)* | 67.08s | 84.41s | 1.50x |
| **Async H2D + `drop_tpu`** *(Recommended, 55 steps)* | **20.75s** | **37.71s** | **3.23x vs. async `d2h` (4.86x vs. sync)** |

> Pre-async `drop_tpu` took **48.96s** (`2.06x`); Run 1 cold switch dropped from `49.00s` to **18.68s** (`35.72s` total).

---

## 3. Quick Start

```bash
git clone https://github.com/OpenMOSS/MOSS-Image.git
cd MOSS-Image
bash tpu/setup-tpu.sh

# 1. Verify TPU device
python3 tpu/run_tpu.py --mode probe

# 2. Base eager generation (auto-enables offload below 40 GiB HBM; Python 3.11–3.14, no JAX required)
python3 tpu/run_tpu.py \
  --mode generate --eager-mode defer_and_fuse \
  --prompt-cache-dir outputs/prompt_cache \
  --height 848 --width 1200 --steps 55 --seed 136879772936584 \
  --prompt "A ceramic teapot on linen" --output outputs/moss_tpu.png
```

<details>
<summary><strong>⚡ Optional: Experimental Pallas GDN Acceleration (requires Python ≥ 3.12 + JAX matching libtpu; preinstalled on Colab TPU)</strong></summary>

```bash
pip install --no-deps deltanet-tpu
python3 tpu/run_tpu.py \
  --mode generate --eager-mode defer_and_fuse --gdn-pallas \
  --prompt-cache-dir outputs/prompt_cache \
  --height 848 --width 1200 --steps 55 --seed 136879772936584 \
  --prompt "A ceramic teapot on linen" --output outputs/moss_tpu.png
```

</details>

Or run [`tpu/moss_image_torchtpu_colab.ipynb`](moss_image_torchtpu_colab.ipynb) directly in Google Colab. Full flags and logs are in [tpu/tpu.md](tpu.md).

---

## 4. Technical Details (Expandable)

<details>
<summary><strong>⚙️ Three-Stage Architecture, drop_tpu, Async H2D & Pallas Bridge</strong></summary>

```text
Prompt text
   │
   ├─► Disk cache (--prompt-cache-dir) / 8-entry CPU LRU ─────────────────┐
   │                                                                      ▼
   └─► Stage 1 [DEFER_NEVER + deltanet-tpu Pallas] (Qwen3.5-9B BF16) ──► CPU embeds [1, L, 4096]
                                                                          │
   Stage 2 [defer_and_fuse] (MOSSImageTransformer BF16, static buckets 512/1024/2048/4096)
     • Precomputed FP32 timestep sinusoid + expanded [1, 1, Q, S] SDPA masks
                                                                          │
                                                                          ▼
   Stage 3 [defer_and_fuse] (AutoencoderKLFlux2 FP32 + NHWC→NCHW GroupNorm hook) ──► PNG
```

- **Three-Stage Runtime (`tpu/tpu_runtime.py`)**: Stage 1 runs unpadded causal encoding (`Qwen3.5-9B`, `15.63 GiB`) and caches `[1, L_true, 4096]` on CPU (`_prompt_cache` LRU + optional `--prompt-cache-dir`). Once loaded, `vae` (`~0.3 GiB`) stays resident across `diffusion <-> text` prompt switches.
- **Zero-D2H `drop_tpu`**: Enabled automatically when host RAM `>= 36 GiB`. Off-stage parameters are released to zero-byte `meta` tensors via `_release_module_to_meta` (`d2h = 0.0s`) and restored from immutable host RAM snapshots. On a `172.95 GiB` host, process RSS after Run 2 is `30.91 GiB` with `138.24 GiB` free.
- **Async `non_blocking=True` H2D**: Batches `.to(tpu, non_blocking=True)` under one `synchronize()`, lifting 400-tensor (`12.50 GiB`) throughput from `0.53 GiB/s` (`23.78s`) to **1.96 GiB/s** (`6.38s`; pre-pinned reaches `2.05 GiB/s` / `6.10s`).
- **Residual `~5.0 GiB` HBM Floor**: Both `d2h` (`5.00–5.03 GiB`) and `drop_tpu` (`5.01–5.03 GiB`) retain a one-time `~5.0 GiB` TorchTPU/PJRT C++ runtime/executable buffer plateau with `0` live Python TPU tensors (`gc.get_objects()`).
- **Pallas GDN & Split Eager (`tpu/tpu_qwen35.py`)**: All 24 `GatedDeltaNet` layers route through `torch.tpu.pallas.jax_op` -> `deltanet_tpu.chunk_gdn` with `128`-aligned static buckets (`0` fallbacks). `--text-eager-mode auto` runs Stage 1 in `default` (`DEFER_NEVER`) + Pallas and Stage 2/3 in `defer_and_fuse`.

</details>

<details>
<summary><strong>🛠️ Hardware Bug Fixes, Encode Microbenchmarks & Unit Tests</strong></summary>

1. **4D Broadcast SDPA Mask `NaN` Fix**: Expands static masks from `[1, 1, 1, S]` to `[1, 1, Q, S]` to prevent `NaN`s at `S >= 512`.
2. **VAE `GroupNorm` Hook**: `ensure_group_norm_contiguous(vae)` converts NHWC convolution outputs to contiguous NCHW before `.view(...)`.
3. **`--compile` (`e0200`) Isolation**: `transformer.compile(backend="tpu", fullgraph=True)` triggers `vmem_address_out_of_range` at the scheduler's `.nonzero()` D2H sync, so `--compile` stays `False` by default.

**Stage-1 `encode-bench` (`329` pos + `24` neg tokens, pre-async H2D)**:

| Mode | H2D | Cold Encode | Neg Encode | Distinct Warm | Repeat Warm | Peak HBM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `eager` (`DEFER_NEVER`) | 28.653s | 1.993s | 0.908s | 1.168s | 0.164s | 15.79 GiB |
| `defer_and_fuse` | 20.577s | 11.023s | 4.659s | 4.901s | 0.072s | 15.73 GiB |
| `eager+gdn_pallas` | 23.860s | **0.310s** | **0.930s** | **0.161s** | 0.092s | 15.83 GiB |
| `defer_and_fuse+gdn_pallas` | 26.311s | 6.672s | 1.684s | 1.859s | **0.049s** | 15.89 GiB |

- **Parity**: TPU HF vs. TPU Pallas image `cos = 0.999442` (`mean_abs = 2.461/255`, `PSNR 32.31 dB`); embedding `cos = 0.999848` (pos) / `0.999758` (neg); TPU HF vs. CUDA `cos = 0.997435`.
- **Checks (32 unit tests)**:
  ```bash
  ruff check tpu gradio_app.py inference_runtime.py tests/test_mps_runtime.py
  python3 -m unittest discover -s tpu/tests -p "test_tpu*.py" -v
  ```

</details>

<details>
<summary><strong>📁 Repository Layout & Next Roadmap (All TPU files organized under `tpu/`)</strong></summary>

| File | Role |
| :--- | :--- |
| [`tpu/__init__.py`](__init__.py) | `tpu` package entrypoint |
| [`tpu/tpu_runtime.py`](tpu_runtime.py) | Staged pipeline, `drop_tpu` + async H2D + resident VAE, static buckets, SDPA/FP32/`GroupNorm` fixes |
| [`tpu/tpu_qwen35.py`](tpu_qwen35.py) | Pallas `chunk_gdn` bridge, `128`-aligned causal padding, persistent prompt cache |
| [`tpu/run_tpu.py`](run_tpu.py) | TPU CLI (`probe`, `encode-only`, `encode-bench`, `generate`, `benchmark`, `stage-switch-bench`) |
| [`tpu/bench_qwen35_gdn_tpu.py`](bench_qwen35_gdn_tpu.py) | Standalone GDN recurrence & single-layer A/B benchmark |
| [`tpu/setup-tpu.sh`](setup-tpu.sh) | TorchTPU 0.2.0 + PyTorch CPU setup script |
| [`tpu/moss_image_torchtpu_colab.ipynb`](moss_image_torchtpu_colab.ipynb) | End-to-end Google Colab notebook |
| [`tpu/tpu.md`](tpu.md) | Bring-up logs, error analyses, and benchmark reports |
| [`tpu/tests/test_tpu_runtime.py`](tests/test_tpu_runtime.py) / [`tpu/tests/test_tpu_qwen35.py`](tests/test_tpu_qwen35.py) | 32 CPU unit & parity tests |
| [`gradio_app.py`](../gradio_app.py) / [`inference_runtime.py`](../inference_runtime.py) | Multi-backend routing entrypoints (CUDA / MPS / TorchTPU) |

**Roadmap**: Pallas `SplashAttention` (>38.4% MFU), scheduler `.nonzero()` fix for `--compile`, CFG `B=2` coalescing + `128/256` buckets, and `Qwen3.5-9B` INT8/W4 quantization for single-device residency.

</details>

---

## 5. Upstream & Citation

Built on **[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** ([Apache-2.0 License](../LICENSE)). See [USERGUIDE.md](../USERGUIDE.md) and [USERGUIDE_ZH.md](../USERGUIDE_ZH.md) for CUDA/MPS/ComfyUI usage.

```bibtex
@misc{ai2026mossimagetechnicalreport,
      title={MOSS-Image Technical Report},
      author={MOSI. AI},
      year={2026},
      eprint={2610.23374},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.23374},
}
```
