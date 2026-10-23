"""Application device placement, optional compilation, and progress reporting."""

from __future__ import annotations

from pathlib import Path

import diffusers
import torch
from huggingface_hub.utils import validate_repo_id

DEFAULT_MODEL = "OpenMOSS-Team/MOSS-Image-Base"


def model_source(model):
    """Resolve local bundles without turning Hub IDs into filesystem paths."""
    model = str(model).strip()
    path = Path(model).expanduser()
    if path.is_dir():
        if not (path / "pipeline_moss_image.py").is_file():
            raise ValueError("The local bundle must include pipeline_moss_image.py from the HF repository.")
        return str(path.resolve())
    if model.startswith(("/", ".", "~")):
        raise FileNotFoundError(f"Model directory does not exist: {model}")
    validate_repo_id(model)
    return model


def load_pipeline(model, local_files_only=False):
    return diffusers.DiffusionPipeline.from_pretrained(
        model_source(model),
        custom_pipeline="pipeline_moss_image",
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
    )


def set_scheduler(pipe, shift):
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, shift=shift)


def cuda_budget_gib():
    if not torch.cuda.is_available():
        return 0.0
    total = torch.cuda.get_device_properties(0).total_memory
    return total * torch.cuda.get_per_process_memory_fraction(0) / 1024**3


def configure_pipeline(pipe, offload="auto", compile_transformer=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if offload == "auto":
        budget = cuda_budget_gib() if device == "cuda" else 0
        offload = "none" if device == "cpu" or budget >= 40 else "hybrid" if budget < 12 else "model"
    if compile_transformer and offload in ("sequential", "hybrid"):
        raise ValueError("Compilation is incompatible with sequential or low-memory offload.")
    if offload == "model":
        pipe.enable_model_cpu_offload(device=device)
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    elif offload == "hybrid":
        if device != "cuda":
            raise ValueError("Low-memory offload requires a CUDA GPU.")
        from memory_offload import enable_hybrid_offload

        enable_hybrid_offload(pipe, torch.device("cuda", torch.cuda.current_device()))
    elif offload == "none":
        pipe.to(device)
    else:
        raise ValueError(f"Unknown offload mode: {offload}")
    if compile_transformer:
        for block in (*pipe.transformer.layers, *pipe.transformer.text_embedder.blocks):
            block.compile(backend="inductor", fullgraph=True, dynamic=True, mode="default")
    return pipe


def generate_with_progress(pipe, report_stage, report_step, **kwargs):
    progress_bar = pipe.progress_bar

    def progress(items):
        report_stage("denoise")
        for index, timestep in enumerate(progress_bar(items)):
            yield timestep
            report_step(pipe, index, timestep, {})
        if kwargs.get("output_type", "pil") != "latent":
            report_stage("decode")

    pipe.progress_bar = progress
    try:
        report_stage("encode")
        return pipe(**kwargs)
    finally:
        pipe.progress_bar = progress_bar
