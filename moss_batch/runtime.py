"""Use the official pipeline with only the components needed by each process."""
from __future__ import annotations

import gc
import importlib.metadata
from pathlib import Path

from moss_batch.cache import load_encoding, save_encoding
from moss_batch.jobs import JobOutput


def resolve_model(model, phase, revision=None, local_model_version=None, local_files_only=False):
    """Pin Hub downloads; local bundles need an explicit portable version label."""
    from huggingface_hub import snapshot_download

    root = Path(model).expanduser()
    if root.is_dir():
        if not local_model_version:
            raise ValueError("Local bundles require --local-model-version; change it whenever the bundle changes.")
        if revision:
            raise ValueError("--revision is for Hub models; use --local-model-version for local bundles.")
        identity = {"kind": "local", "version": local_model_version}
    else:
        if local_model_version:
            raise ValueError("--local-model-version is only for a local model directory.")
        components = ("text_encoder", "tokenizer") if phase == "encode" else ("transformer", "scheduler", "vae")
        root = Path(snapshot_download(
            model, revision=revision, local_files_only=local_files_only,
            allow_patterns=["*.py", "model_index.json", *[f"{part}/*" for part in components]],
        ))
        identity = {"kind": "hub", "repo_id": model, "revision": root.name}
    if not (root / "pipeline_moss_image.py").is_file():
        raise ValueError("Model bundle must include the official pipeline_moss_image.py.")
    return root.resolve(), identity


def load_stage(source, phase, offload="model", device="cuda"):
    import torch
    from diffusers.utils.dynamic_modules_utils import get_class_from_dynamic_module

    if phase not in ("encode", "generate"):
        raise ValueError("Unknown inference stage.")
    if offload not in ("none", "model", "sequential"):
        raise ValueError("Unknown offload mode.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu --offload none for CPU execution.")
    if device == "cpu" and offload != "none":
        raise ValueError("CPU execution requires --offload none.")
    omitted = dict(transformer=None, scheduler=None, vae=None) if phase == "encode" else dict(
        text_encoder=None, tokenizer=None,
    )
    # Call the concrete class: Diffusers 0.38's base loader otherwise ignores
    # explicit None for this custom pipeline's optional constructor components.
    pipeline_class = get_class_from_dynamic_module(
        str(source), module_file="pipeline_moss_image.py", class_name="MOSSImagePipeline",
        local_files_only=True, trust_remote_code=True,
    )
    pipe = pipeline_class.from_pretrained(
        str(source),
        local_files_only=True, trust_remote_code=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
        **omitted,
    )
    if offload == "model":
        pipe.enable_model_cpu_offload(device=device)
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        pipe.to(device)
    return pipe


def release(pipe):
    import torch

    if pipe is not None:
        pipe.remove_all_hooks()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encoding_request(job, identity):
    return {
        "source": identity, "prompt": job.prompt, "negative_prompt": job.negative_prompt,
        "max_sequence_length": job.parameters["max_sequence_length"], "dtype": "bfloat16",
    }


def _prompt_key(job):
    return job.prompt, job.negative_prompt, job.parameters["max_sequence_length"]


def _check_encoding_dtype(tensors):
    import torch

    for name in ("prompt_embeds", "negative_prompt_embeds"):
        if name in tensors and tensors[name].dtype != torch.bfloat16:
            raise ValueError(f"{name} must match the recorded bfloat16 encoding dtype.")


def _load_encoding(path, expected):
    tensors, metadata = load_encoding(path, expected=expected)
    _check_encoding_dtype(tensors)
    return tensors, metadata


def encode_jobs(jobs, source, identity, cache_dir, offload="model", device="cuda", report=print):
    """Encode both branches, including CFG=1 jobs, so later CFG changes can reuse them."""
    pending = []
    saved = {}
    for job in jobs:
        path = Path(cache_dir) / job.id
        expected = encoding_request(job, identity)
        if path.exists():
            _load_encoding(path, expected)
            saved[_prompt_key(job)] = path
            report(f"{job.id}: encoding already saved")
        else:
            pending.append((job, path, expected))
    if not pending:
        return
    pipe = None
    try:
        for job, path, metadata in pending:
            # Reuse on-disk encodings without retaining every prompt in memory.
            key = _prompt_key(job)
            if key in saved:
                tensors, metadata = _load_encoding(saved[key], metadata)
            else:
                if pipe is None:
                    pipe = load_stage(source, "encode", offload, device)
                positive, positive_mask = pipe.encode_prompt(job.prompt, key[2])
                negative, negative_mask = pipe.encode_prompt(job.negative_prompt, key[2])
                tensors = {
                    "prompt_embeds": positive.cpu().contiguous(),
                    "prompt_attention_mask": positive_mask.cpu().contiguous(),
                    "negative_prompt_embeds": negative.cpu().contiguous(),
                    "negative_prompt_attention_mask": negative_mask.cpu().contiguous(),
                }
                del positive, positive_mask, negative, negative_mask
                pipe.maybe_free_model_hooks()
                metadata = {
                    **metadata,
                    "text_encoder_class": type(pipe.text_encoder).__name__,
                    "tokenizer_class": type(pipe.tokenizer).__name__,
                    "chat_template": pipe.tokenizer.chat_template,
                    "versions": {
                        name: importlib.metadata.version(name) for name in ("torch", "transformers", "diffusers")
                    },
                }
            _check_encoding_dtype(tensors)
            save_encoding(path, tensors, metadata)
            saved[key] = path
            report(f"{job.id}: encoding saved")
    finally:
        release(pipe)


def generate_jobs(jobs, source, identity, cache_dir, output_dir, offload="model", device="cuda", report=print):
    """Sequential generation; completed jobs and actual random seeds survive restarts."""
    import torch

    pipe = None
    try:
        for job in jobs:
            tensors, metadata = _load_encoding(Path(cache_dir) / job.id, encoding_request(job, identity))
            provenance = {"source": identity, "encoding": metadata}
            output = JobOutput(output_dir, job, provenance)
            seed = output.prepare()
            if output.is_complete():
                report(f"{job.id}: already complete (seed {seed})")
                continue
            if job.parameters["guidance_scale"] != 1 and "negative_prompt_embeds" not in tensors:
                raise ValueError(f"{job.id}: CFG requires negative prompt encodings; encode both branches again.")
            if pipe is None:
                pipe = load_stage(source, "generate", offload, device)
            parameters = dict(job.parameters)
            shift = parameters.pop("shift")
            pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, shift=shift)
            depth = pipe.transformer.config.num_layers
            if any(layer >= depth for layer in parameters["uncond_skip_layers"] + parameters["slg_layers"]):
                raise ValueError(f"{job.id}: skip layer indices must be below {depth}.")
            result = pipe(
                **tensors, **parameters, generator=torch.Generator("cpu").manual_seed(seed),
                num_images_per_prompt=1,
            )
            output.complete(result.images[0])
            report(f"{job.id}: saved {output.image_path} (seed {seed})")
    finally:
        release(pipe)
