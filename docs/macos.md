# Experimental Apple silicon inference (MPS)

[简体中文](macos_zh.md)

This optional path runs MOSS-Image through PyTorch Metal/MPS. Recorded real-weight validation
is on an **M1 Max / 32 GB unified-memory Mac**, with SFT. Other chips, lower-memory systems,
Base and older dependency versions need testing. No quantization is applied.

## Install and launch

Use native arm64 Python 3.12 and a compatible macOS/PyTorch installation. Allow about 40 GB
free disk for one full model bundle (~25 GB), the environment and download files.
From the cloned repository root:

```bash
MOSS_PYTHON=python3.12 bash scripts/setup-mac.sh
.venv/bin/hf download OpenMOSS-Team/MOSS-Image-SFT --local-dir models/MOSS-Image-SFT
bash scripts/gradio-mac.sh
```

Open http://127.0.0.1:7860. Start with 512×512, 10 steps and one image. Keep the terminal
running; loading and inference can take minutes. Original PNGs and parameters.json are saved
under `outputs/gradio/`. Stop with Ctrl+C. Subsequent runs only need the launch command.
The Mac launcher uses local files only and disables Gradio analytics.

`constraints-mac.txt` lists the versions used for the recorded local runs: torch 2.14.1,
diffusers 0.41.0, transformers 5.20.1, accelerate 1.15.0 and Gradio 6.30.0. These constraints
apply to the optional Mac setup script, not to the existing installation path.

The launcher defaults the DiT to FP16. To compare BF16:

```bash
MOSS_MPS_DIT_DTYPE=bfloat16 bash scripts/gradio-mac.sh
```

The text encoder remains BF16 and the VAE FP32. Direct `mps_runtime.load_pipeline` defaults
to BF16 unless `MOSS_MPS_DIT_DTYPE` is set before import. CUDA offload modes and Compile DiT
do not apply to this staged MPS path; the UI disables those controls on MPS.

## Implementation and review points

The Qwen encoder (~17 GB) and DiT (~10 GB) are staged so only one is resident on MPS at a
time. Positive and negative prompts are encoded before the DiT is loaded. An eight-entry
LRU prompt cache permits reuse when only seed, dimensions or sampling settings change.

The downloaded custom pipeline computes a timestep sinusoid in FP64, unsupported by MPS.
The small sinusoid operation is moved to CPU at FP64 and its result copied back. Because
the standalone pipeline is distributed in the model bundle, this currently uses a narrowly
matched source replacement and stops explicitly if that source changes. The ComfyUI model
has the equivalent conditional CPU calculation; the full ComfyUI path was not validated here.

A locally observed loading deadlock involved pending Metal copies and cache/synchronization
calls holding the GIL. The current workaround drains copies through a CPU transfer before
calling those methods. It wraps `torch.mps.empty_cache` and `torch.mps.synchronize` when MPS
is available, including other callers in the process; that process-wide effect is an explicit
review point for this experimental implementation.

## Evidence and limits

Recorded SFT calls on M1 Max / 32 GB:

| Output | Settings | Full recorded generation call |
| --- | --- | --- |
| Chinese chalkboard | 1024×1024, 20 steps, BF16, seed 42 | 320.362 s |
| English character posters | 768×1024, 28 steps, FP16 | 395.150 / 367.596 / 344.072 s |

[Images, original metadata and prompts](https://github.com/EdwinjJ1/moss-image-mac/tree/main/assets/samples)
are available as validation evidence. These are end-to-end records, not controlled throughput,
precision-equivalence or peak-memory measurements. CUDA execution is retained but was not
hardware-regression-tested by this Mac contribution.

CPU regression checks, without downloading weights:

```bash
python -m pip install -e '.[ui]' -c constraints-mac.txt
python -m unittest discover -s tests -p test_mps_runtime.py -v
```

The checks cover staging order, prompt cache eviction, CPU sinusoid arithmetic, source-change
failure, MPS routing and UI construction. Real MPS generation remains a separate hardware check.
If MPS is unavailable, verify native arm64 Python and macOS compatibility. For memory pressure,
reduce resolution/count and close heavy applications. For a source-patch mismatch, report the
model revision and error rather than bypassing the check.
