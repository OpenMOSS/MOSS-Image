"""Apple-silicon (MPS) loading that keeps one large model on the GPU at a time.

The Qwen text encoder (~17 GB) and the DiT (~10 GB) do not fit in a 32 GB Mac's
GPU working set together. The text encoder is loaded from disk only to encode a
new prompt and is released before sampling; the DiT then comes back. Prompt
embeddings are cached, so a new seed, size or step count skips the text encoder.
"""
from __future__ import annotations

import gc
import importlib.util
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

DEVICE = "mps"
CACHED_PROMPTS = 8
# MOSS_MPS_DIT_DTYPE=float16 runs the DiT in fp16, which M1/M2 GPUs compute natively.
DIT_DTYPE = getattr(torch, os.environ.get("MOSS_MPS_DIT_DTYPE", "bfloat16"), None)
if DIT_DTYPE not in (torch.float16, torch.bfloat16):
    raise ValueError("MOSS_MPS_DIT_DTYPE must be float16 or bfloat16.")

# MPS has no float64. The DiT computes its timestep sinusoid in float64, so
# that one calculation runs on the CPU with the same precision.
_FLOAT64_TIMESTEP = (
    "        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)\n"
    "        phase = times.double()[:, None] * freq\n"
    "        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()\n"
)
_CPU_TIMESTEP = (
    "        freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)\n"
    "        phase = times.cpu().double()[:, None] * freq\n"
    "        embedding = torch.cat((phase.cos(), phase.sin()), -1).float().to(times.device)\n"
)


def _wait_without_gil(original):
    # torch.mps.empty_cache() and synchronize() wait on the GPU while holding the GIL.
    # After a non_blocking copy from a CPU tensor whose storage belongs to a Python object
    # (safetensors, frombuffer), Metal's completion handler needs the GIL to release it, so
    # the two deadlock; Diffusers does exactly this when loading with device_map.
    # Copying a value back to the CPU waits with the GIL released and drains them first.
    def wait(*args, **kwargs):
        torch.zeros(1, device=DEVICE).cpu()
        return original(*args, **kwargs)
    return wait


if torch.backends.mps.is_available():
    torch.mps.empty_cache = _wait_without_gil(torch.mps.empty_cache)
    torch.mps.synchronize = _wait_without_gil(torch.mps.synchronize)


def is_available():
    return not torch.cuda.is_available() and torch.backends.mps.is_available()


def empty_cache():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _bundle_root(model, local_files_only):
    path = Path(model)
    if path.is_dir():
        return path
    return Path(snapshot_download(model, local_files_only=local_files_only))


def _import_pipeline_module(root):
    path = root / "pipeline_moss_image.py"
    source = path.read_text()
    if _FLOAT64_TIMESTEP not in source:
        raise ValueError(f"{path} changed; update the float64 patch in mps_runtime.py.")
    name = "pipeline_moss_image_mps"
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None, origin=str(path)))
    sys.modules[name] = module
    exec(compile(source.replace(_FLOAT64_TIMESTEP, _CPU_TIMESTEP), str(path), "exec"), module.__dict__)
    return module


def _staged_pipeline_class(module):
    class StagedMOSSImagePipeline(module.MOSSImagePipeline):
        """Swaps the text encoder and the DiT on the GPU instead of holding both."""

        @property
        def _execution_device(self):
            return torch.device(DEVICE)

        def _release(self, name):
            if getattr(self, name) is not None:
                setattr(self, name, None)
                empty_cache()

        def use_text_encoder(self):
            if self.text_encoder is None:
                self._release("transformer")
                self.text_encoder = AutoModel.from_pretrained(
                    self.root / "text_encoder", dtype=torch.bfloat16, device_map=DEVICE
                ).eval()

        def use_transformer(self):
            if self.transformer is None:
                self._release("text_encoder")
                self.transformer = module.MOSSImageTransformer.from_pretrained(
                    self.root / "transformer", torch_dtype=DIT_DTYPE, device_map=DEVICE
                ).eval()

        def encode_prompt(self, prompt, max_sequence_length=4096):
            key = ((prompt,) if isinstance(prompt, str) else tuple(prompt), max_sequence_length)
            if key not in self.embeddings:
                self.use_text_encoder()
                # An autograd graph would keep the released text encoder alive in the cache.
                with torch.no_grad():
                    self.embeddings[key] = super().encode_prompt(prompt, max_sequence_length)
                while len(self.embeddings) > CACHED_PROMPTS:
                    self.embeddings.popitem(last=False)
            self.embeddings.move_to_end(key)
            return self.embeddings[key]

        def __call__(self, prompt=None, *, negative_prompt="", guidance_scale=3.0, max_sequence_length=4096,
                     prompt_embeds=None, prompt_attention_mask=None,
                     negative_prompt_embeds=None, negative_prompt_attention_mask=None, **kwargs):
            # Encode both prompts before the DiT returns to the GPU.
            if prompt is not None and prompt_embeds is None:
                prompt_embeds, prompt_attention_mask = self.encode_prompt(prompt, max_sequence_length)
                prompt = None
            if guidance_scale != 1.0 and negative_prompt_embeds is None and prompt_embeds is not None:
                batch = prompt_embeds.shape[0]
                negatives = [negative_prompt] * batch if isinstance(negative_prompt, str) else negative_prompt
                negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                    negatives, max_sequence_length
                )
            self.use_transformer()
            return super().__call__(
                prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                prompt_embeds=prompt_embeds, prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask, **kwargs,
            )

    return StagedMOSSImagePipeline


def load_pipeline(model, local_files_only=False):
    root = _bundle_root(model, local_files_only)
    module = _import_pipeline_module(root)
    pipe = _staged_pipeline_class(module)(
        transformer=None,
        scheduler=module.MOSSImageScheduler.from_pretrained(root / "scheduler"),
        tokenizer=AutoTokenizer.from_pretrained(root / "tokenizer"),
        vae=AutoencoderKLFlux2.from_pretrained(root / "vae", torch_dtype=torch.float32).to(DEVICE).eval(),
    )
    pipe.root = root
    # The pipeline class is defined here; provenance records the bundle source it patches and runs.
    pipe.pipeline_file = root / "pipeline_moss_image.py"
    pipe.embeddings = OrderedDict()
    pipe.use_transformer()
    return pipe
