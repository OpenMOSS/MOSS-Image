"""Application device placement, optional compilation, and progress reporting."""
from __future__ import annotations

import inspect
import sys
import textwrap
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


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cuda_budget_gib():
    if not torch.cuda.is_available():
        return 0.0
    total = torch.cuda.get_device_properties(0).total_memory
    return total * torch.cuda.get_per_process_memory_fraction(0) / 1024**3


def empty_device_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def is_out_of_memory(exc):
    # MPS reports exhaustion as a plain RuntimeError rather than torch.OutOfMemoryError.
    return isinstance(exc, torch.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def patch_transformer_for_mps(transformer):
    """MPS has no float64; build the timestep sinusoid on CPU with the same math, then move it."""
    cls = type(transformer)
    if getattr(cls, "_mps_patched", False):
        return
    source = textwrap.dedent(inspect.getsource(cls.forward))
    old = (
        "freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)\n"
        "    phase = times.double()[:, None] * freq\n"
        "    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()"
    )
    new = (
        "freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)\n"
        "    phase = times.cpu().double()[:, None] * freq\n"
        "    embedding = torch.cat((phase.cos(), phase.sin()), -1).float().to(times.device)"
    )
    if source.count(old) != 1:
        raise RuntimeError("MOSSImageTransformer.forward changed upstream; the MPS float64 patch no longer applies.")
    namespace = {}
    code = compile(source.replace(old, new), inspect.getsourcefile(cls), "exec")
    exec(code, vars(sys.modules[cls.__module__]), namespace)
    cls.forward = namespace["forward"]
    cls._mps_patched = True


def drop_vision_tower(pipe):
    """The Qwen3.5 vision tower (~0.9 GB) only runs for image inputs, which MOSS-Image never passes."""
    if getattr(pipe.text_encoder, "visual", None) is not None:
        pipe.text_encoder.visual = None


def configure_pipeline(pipe, offload="auto", compile_transformer=False):
    device = default_device()
    if device == "mps":
        patch_transformer_for_mps(pipe.transformer)
        drop_vision_tower(pipe)
    if offload == "auto":
        # Unified memory: CPU offload only copies within the same RAM, so keep everything resident.
        offload = "none" if device in ("cpu", "mps") or cuda_budget_gib() >= 40 else "model"
    if compile_transformer and offload == "sequential":
        raise ValueError("Compilation is incompatible with sequential offload.")
    if offload == "model":
        pipe.enable_model_cpu_offload(device=device)
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
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
