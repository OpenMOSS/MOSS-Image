"""Keep DiT weights in RAM and load text encoder/VAE weights on demand."""

from pathlib import Path

from accelerate import cpu_offload
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, attach_align_device_hook
from accelerate.utils import OffloadedWeightsLoader
from huggingface_hub import snapshot_download
from safetensors import safe_open


def _file_weights(model, folder):
    """Index a loaded component without retaining its CPU parameter tensors."""
    index = {}
    shapes = {}
    for file in sorted(folder.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                if name in index:
                    raise ValueError(f"Duplicate tensor {name!r} in {folder}")
                index[name] = {"safetensors_file": str(file), "weight_name": name}
                shapes[name] = tuple(tensors.get_slice(name).get_shape())
    for name, parameter in model.named_parameters():
        if name not in index or shapes[name] != tuple(parameter.shape):
            raise ValueError(f"Missing or incompatible checkpoint tensor: {folder.name}.{name}")
        # Preserve the dtype selected by from_pretrained (including FP32 norms).
        index[name]["dtype"] = str(parameter.dtype).removeprefix("torch.")
    return OffloadedWeightsLoader(index=index, device="cpu")


def _offload_from_files(model, weights, device):
    add_hook_to_module(model, AlignDevicesHook(io_same_device=True), append=True)
    attach_align_device_hook(
        model,
        execution_device=device,
        offload=True,
        offload_buffers=False,
        weights_map=weights,
    )


def enable_hybrid_offload(pipe, device):
    """Keep DiT in CPU RAM and offload TE/VAE to the original checkpoint files."""
    source = pipe.config._name_or_path
    folder = Path(source).expanduser()
    if not folder.is_dir():
        folder = Path(snapshot_download(source, local_files_only=True))
    # Validate both components before installing any hooks.
    weights = {name: _file_weights(getattr(pipe, name), folder / name) for name in ("text_encoder", "vae")}
    pipe.remove_all_hooks()
    for name, state in weights.items():
        _offload_from_files(getattr(pipe, name), state, device)
    # A CPU .to() alone may retain an mmap view and re-read DiT from disk when
    # RAM is tight. Clone once so sampling uses an in-memory copy instead.
    state = {name: value.detach().cpu().clone() for name, value in pipe.transformer.state_dict().items()}
    cpu_offload(pipe.transformer, execution_device=device, state_dict=state, offload_buffers=False)
    return pipe
