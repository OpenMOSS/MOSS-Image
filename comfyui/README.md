# MOSS-Image for ComfyUI

English | [简体中文](README_ZH.md)

MOSS-Image supports loading directly from a local Hugging Face / Diffusers model
directory. Its standard ComfyUI nodes output MODEL, CLIP and VAE, using native
sampling, SLG and memory management.

## Native ComfyUI support (recommended)

[MOSS-Image support](https://github.com/Comfy-Org/ComfyUI/pull/17095) was merged into ComfyUI on October 16, 2026. Use ComfyUI master/nightly containing commit `e087d8f6` or later and install its current `requirements.txt` dependencies; this path does not require the repository plugin. As of October 19, 2026, the `v0.41.3` release does not include this support.

Use the [ComfyUI-format models and workflows](https://huggingface.co/Comfy-Org/MOSS-Image): place the diffusion model in `models/diffusion_models/`, the text encoder in `models/text_encoders/`, and the VAE in `models/vae/`. Use **Load Diffusion Model**, **CLIP Loader** with type **moss_image**, and **Load VAE**. The built-in model defaults to shift 5 and includes a **MOSS-Image Layout Prompt** node.

Existing plugin workflows use different loader nodes. Import a native workflow when switching; swapping only the model loader in the old workflow is insufficient. In particular, the native model uses a different timestep multiplier, so do not carry over the plugin workflow's **ModelSamplingSD3** node unchanged. To adjust native model shift, use **ModelSamplingAuraFlow** (multiplier 1).

## HF directory plugin (optional)

The instructions below cover the repository plugin and its workflows. Keep it if you want to load a local Hugging Face / Diffusers directory directly or continue using an existing plugin workflow.

### Installation

Copy `comfyui/moss-image/` to `ComfyUI/custom_nodes/moss-image/` and restart
ComfyUI. The native plugin uses ComfyUI's dependencies and can be installed
independently of this repository.

Alternatively, create links from the repository root:

```bash
python comfyui/install.py --comfy-root /path/to/ComfyUI
```

For a separate local ComfyUI installation, use the [local launch scripts](#local-launch) below.

Existing user workflows are not overwritten. Environments, weights, generated
images and user settings stay local and are excluded from version control.

## Workflow

1. Download [moss-image.json](moss-image/example_workflows/moss-image.json) and drag it into ComfyUI.
2. Download [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base). In **MOSS-Image HF Loader**, set `model_path` to the downloaded local directory containing `transformer/`, `text_encoder/`, `tokenizer/`, and `vae/`.
3. Edit the positive and negative prompts, then click **Run** to generate an image.

The workflow is ready to run with **50 steps, CFG 3, Shift 5, and negative-branch
skipping at layer 22**, using Euler / sgm_uniform at 1024×1024.
See the [user guide](../USERGUIDE.md) for prompt and sampling recommendations.

## Local launch

```bash
bash comfyui/local/setup.sh
bash comfyui/local/run.sh
```

The setup script clones ComfyUI into the Git-ignored `local/ComfyUI/` directory
if it is missing, creates a separate environment, and installs its dependencies.
It installs the nodes and the MOSS-Image workflow without
overwriting an existing file of the same name. Set `MOSS_PYTHON` to choose the
interpreter and `UV_INDEX_URL` to choose the package index.

`COMFYUI_ROOT` selects an existing ComfyUI checkout. `COMFY_HOST` and
`COMFY_PORT` set the listening address; `CUDA_VISIBLE_DEVICES` selects the GPU.
Launch defaults enable FP32 VAE and Latent2RGB previews and disable image
metadata. The runtime's `local/env.sh` can configure launch options.

## Advanced settings

The [API JSON](moss-image/example_workflows/moss-image-api.json) provides the same
workflow for programmatic submission.

The loader reads local single-file or sharded safetensors weights; it does not
download models from a Hub repository ID. `weight_dtype` controls DiT precision,
`text_device` selects the text encoder's device, and the VAE uses FP32.
Text encoding is capped at 4096 tokens.

Clear Simple SLG's `single_layers` to disable negative-branch skipping. To use
standard SLG instead, replace Simple SLG with `SkipLayerGuidanceDiT`, choose
`single_layers`, and set scale 3, start 0.01, end 1, and rescaling 0. Keep
`double_layers` empty. The user guide does not specify standard SLG layer indices.

Keep ModelSamplingSD3 before SLG so scheduling and SLG thresholds share the same
shift. ComfyUI SLG intervals use sigma-based percentages. Its native scheduler
does not use the standalone pipeline's training-grid snapping, so matching seeds
and parameters do not guarantee identical images.

Separate component loaders remain available for custom workflows. LoRAs with
matching model keys can use the built-in LoRA loader. For SLG or changing block
patches, use eager execution: TorchCompileModel filters
transformer-option guards.
