from __future__ import annotations

import gc
import hashlib
import importlib.util
import sys
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from threading import RLock
from time import perf_counter as _perf_counter

import torch
from huggingface_hub import snapshot_download

from inference_runtime import model_source
from tpu import tpu_qwen35

CACHED_PROMPTS = 8
DEFAULT_PROMPT_BUCKETS = (512, 1024, 2048, 4096)
TPU_BACKEND = "tpu"
VALID_EAGER_MODES = ("default", "defer_and_fuse")
VALID_MATMUL_PRECISIONS = ("highest", "high", "medium")
VALID_OFFLOAD_MODES = ("auto", "model", "staged", "none")
VALID_STAGE_SWITCH_MODES = ("auto", "drop_tpu", "d2h")
DEFAULT_DROP_TPU_MIN_HOST_GIB = 36.0
TEXT_COMPONENTS = ("text_encoder", "tokenizer")
DIFFUSION_COMPONENTS = ("transformer", "vae", "scheduler")

_TRIED_TPU_IMPORT = False
_PATCHED_MODULES: dict[str, tuple[str, object]] = {}


def _ensure_tpu_backend_loaded():
    global _TRIED_TPU_IMPORT
    if _TRIED_TPU_IMPORT:
        return
    _TRIED_TPU_IMPORT = True
    try:
        import torch_tpu  # noqa: F401
    except ImportError:
        pass


def is_available() -> bool:
    if torch.cuda.is_available():
        return False
    _ensure_tpu_backend_loaded()
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and callable(getattr(accelerator, "is_available", None)):
        try:
            if accelerator.is_available():
                current = accelerator.current_accelerator()
                if current is not None and getattr(current, "type", None) == "tpu":
                    return True
        except RuntimeError:
            pass
    tpu_mod = getattr(torch, "tpu", None)
    tpu_is_available = getattr(tpu_mod, "is_available", None)
    if callable(tpu_is_available):
        return bool(tpu_is_available())
    return False


def tpu_device() -> torch.device:
    return torch.device("tpu")


def resolve_model_bundle(model, *, local_files_only=False, components=TEXT_COMPONENTS + DIFFUSION_COMPONENTS):
    source = model_source(model)
    path = Path(source)
    if path.is_dir():
        return path
    return Path(snapshot_download(
        source,
        local_files_only=local_files_only,
        allow_patterns=["pipeline_moss_image.py", "model_index.json", *(f"{name}/*" for name in components)],
    ))


def tpu_budget_gib() -> float:
    if not is_available():
        return 0.0
    accelerator = getattr(torch, "accelerator", None)
    get_memory_info = getattr(accelerator, "get_memory_info", None)
    if callable(get_memory_info):
        try:
            _free, total = get_memory_info()
            return float(total) / 1024**3
        except Exception:
            return 0.0
    return 0.0


def memory_allocated_gib() -> float | None:
    if not is_available():
        return None
    accelerator = getattr(torch, "accelerator", None)
    fn = getattr(accelerator, "memory_allocated", None)
    if callable(fn):
        try:
            return float(fn()) / 1024**3
        except Exception:
            return None
    return None


def max_memory_allocated_gib() -> float | None:
    if not is_available():
        return None
    accelerator = getattr(torch, "accelerator", None)
    fn = getattr(accelerator, "max_memory_allocated", None)
    if callable(fn):
        try:
            return float(fn()) / 1024**3
        except Exception:
            return None
    return None


def host_memory_info_gib() -> dict[str, float | None]:
    """Return Linux host RAM total/available and process RSS in GiB, or None fields when unavailable."""
    info: dict[str, float | None] = {
        "total_gib": None,
        "available_gib": None,
        "process_rss_gib": None,
    }
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 2:
                    continue
                if parts[0] == "MemTotal:":
                    info["total_gib"] = round(int(parts[1]) / (1024**2), 3)
                elif parts[0] == "MemAvailable:":
                    info["available_gib"] = round(int(parts[1]) / (1024**2), 3)
        except Exception:
            pass
    status = Path("/proc/self/status")
    if status.is_file():
        try:
            for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        info["process_rss_gib"] = round(int(parts[1]) / (1024**2), 3)
                    break
        except Exception:
            pass
    return info


def reset_peak_memory_stats() -> None:
    if not is_available():
        return
    accelerator = getattr(torch, "accelerator", None)
    fn = getattr(accelerator, "reset_peak_memory_stats", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


def device_name() -> str | None:
    if not is_available():
        return None
    tpu_mod = getattr(torch, "tpu", None)
    get_name = getattr(tpu_mod, "get_device_name", None)
    if callable(get_name):
        try:
            name = get_name(0)
            if name:
                return str(name)
        except Exception:
            pass
    try:
        import torch_tpu

        get_pkg_name = getattr(torch_tpu, "get_device_name", None)
        if callable(get_pkg_name):
            name = get_pkg_name()
            if name:
                return str(name)
    except Exception:
        pass
    budget = tpu_budget_gib()
    if budget > 0:
        return f"Google Cloud TPU ({budget:.0f} GiB HBM)"
    return "Google Cloud TPU"


def synchronize() -> None:
    if not is_available():
        return
    accelerator = getattr(torch, "accelerator", None)
    sync = getattr(accelerator, "synchronize", None)
    if callable(sync):
        sync()
        return
    tpu_mod = getattr(torch, "tpu", None)
    legacy_sync = getattr(tpu_mod, "synchronize", None)
    if callable(legacy_sync):
        legacy_sync()


def empty_cache() -> None:
    if not is_available():
        return
    synchronize()
    accelerator = getattr(torch, "accelerator", None)
    fn = getattr(accelerator, "empty_cache", None)
    if callable(fn):
        fn()


def validate_eager_mode(mode: str, *, label: str = "TPU eager mode") -> str:
    if mode not in VALID_EAGER_MODES:
        raise ValueError(f"Unknown {label}: {mode}. Expected one of {VALID_EAGER_MODES}.")
    return mode


def configure_eager_mode(mode: str, device: torch.device):
    validate_eager_mode(mode)
    if device.type != "tpu":
        return nullcontext()
    eager_mode = torch.tpu.EagerMode
    target = eager_mode.DEFER_AND_FUSE if mode == "defer_and_fuse" else eager_mode.DEFER_NEVER
    return torch.tpu.set_eager_mode(target)


def configure_precision(matmul_precision: str = "high") -> str:
    if matmul_precision not in VALID_MATMUL_PRECISIONS:
        raise ValueError(
            f"Unknown float32 matmul precision: {matmul_precision}. "
            f"Expected one of {VALID_MATMUL_PRECISIONS}."
        )
    set_precision = getattr(torch, "set_float32_matmul_precision", None)
    if callable(set_precision):
        set_precision(matmul_precision)
    return matmul_precision


def validate_stage_switch_mode(mode: str, *, label: str = "TPU stage switch mode") -> str:
    if mode not in VALID_STAGE_SWITCH_MODES:
        raise ValueError(f"Unknown {label}: {mode}. Expected one of {VALID_STAGE_SWITCH_MODES}.")
    return mode


def resolve_stage_switch_mode(
    mode: str,
    device: torch.device,
    *,
    host_total_gib: float | None = None,
    min_host_gib: float = DEFAULT_DROP_TPU_MIN_HOST_GIB,
    force_drop_tpu: bool = False,
) -> str:
    """Resolve 'auto' stage switch mode to 'drop_tpu' (skip D2H) or 'd2h'."""
    validated = validate_stage_switch_mode(mode)
    if device.type == "cpu" and not force_drop_tpu:
        return "d2h"
    if validated != "auto":
        return validated
    total_gib = host_memory_info_gib()["total_gib"] if host_total_gib is None else host_total_gib
    if total_gib is None or total_gib >= min_host_gib:
        return "drop_tpu"
    return "d2h"


def _snapshot_module_cpu_weights(module: torch.nn.Module) -> dict[str, list]:
    """Capture references to frozen CPU parameters and buffers (copying only if not on CPU)."""
    param_entries: list[tuple[torch.nn.Module, str, int, bool, torch.Tensor]] = []
    cached_cpu_params: dict[int, torch.Tensor] = {}
    buffer_entries: list[tuple[torch.nn.Module, str, torch.Tensor]] = []
    cached_cpu_buffers: dict[int, torch.Tensor] = {}

    for submod in module.modules():
        for pname, param in submod._parameters.items():
            if param is None:
                continue
            pid = id(param)
            cpu_tensor = cached_cpu_params.get(pid)
            if cpu_tensor is None:
                data = param.data
                if data.is_meta:
                    raise ValueError(f"Cannot snapshot parameter {pname!r} from a meta tensor.")
                cpu_tensor = data.detach() if data.device.type == "cpu" else data.detach().to("cpu")
                cached_cpu_params[pid] = cpu_tensor
            param_entries.append((submod, pname, pid, bool(param.requires_grad), cpu_tensor))

        for buf_name, buf in submod._buffers.items():
            if buf is None:
                continue
            bid = id(buf)
            cpu_buf = cached_cpu_buffers.get(bid)
            if cpu_buf is None:
                if buf.is_meta:
                    raise ValueError(f"Cannot snapshot meta buffer {buf_name!r} without CPU data.")
                cpu_buf = buf.detach() if buf.device.type == "cpu" else buf.detach().to("cpu")
                cached_cpu_buffers[bid] = cpu_buf
            buffer_entries.append((submod, buf_name, cpu_buf))

    return {"params": param_entries, "buffers": buffer_entries}


def _release_module_to_meta(module: torch.nn.Module) -> None:
    """Release device parameter/buffer storage onto zero-byte meta tensors without D2H copy."""
    meta_params: dict[int, torch.nn.Parameter] = {}
    meta_buffers: dict[int, torch.Tensor] = {}
    for submod in module.modules():
        for pname, param in list(submod._parameters.items()):
            if param is None:
                continue
            pid = id(param)
            meta_param = meta_params.get(pid)
            if meta_param is None:
                meta_param = torch.nn.Parameter(
                    torch.empty(param.shape, dtype=param.dtype, device="meta"),
                    requires_grad=param.requires_grad,
                )
                meta_params[pid] = meta_param
            submod._parameters[pname] = meta_param
        for buf_name, buf in list(submod._buffers.items()):
            if buf is None:
                continue
            bid = id(buf)
            meta_buf = meta_buffers.get(bid)
            if meta_buf is None:
                meta_buf = torch.empty(buf.shape, dtype=buf.dtype, device="meta")
                meta_buffers[bid] = meta_buf
            submod._buffers[buf_name] = meta_buf


def _restore_module_from_cpu_snapshot(
    module: torch.nn.Module,
    snapshot: dict[str, list],
    target_device: torch.device,
) -> None:
    """Restore parameters and buffers onto target_device from a retained CPU snapshot."""
    del module
    use_non_blocking = target_device.type != "cpu"
    converted_tensors: dict[int, torch.Tensor] = {}
    restored_params: dict[int, torch.nn.Parameter] = {}

    def _move_once(cpu_tensor: torch.Tensor) -> torch.Tensor:
        if cpu_tensor.device == target_device:
            return cpu_tensor
        tid = id(cpu_tensor)
        moved = converted_tensors.get(tid)
        if moved is None:
            moved = cpu_tensor.to(target_device, non_blocking=use_non_blocking)
            converted_tensors[tid] = moved
        return moved

    for submod, pname, pid, req_grad, cpu_tensor in snapshot["params"]:
        restored = restored_params.get(pid)
        if restored is None:
            restored = torch.nn.Parameter(_move_once(cpu_tensor), requires_grad=req_grad)
            restored_params[pid] = restored
        submod._parameters[pname] = restored

    for submod, buf_name, cpu_buf in snapshot["buffers"]:
        submod._buffers[buf_name] = _move_once(cpu_buf)


def select_prompt_bucket(
    seq_len: int,
    buckets: tuple[int, ...] | list[int] = DEFAULT_PROMPT_BUCKETS,
    max_length: int = 4096,
) -> int:
    if seq_len < 1:
        raise ValueError("seq_len must be positive.")
    if max_length < 1:
        raise ValueError("max_length must be positive.")
    if seq_len > max_length:
        raise ValueError(f"seq_len ({seq_len}) exceeds max_length ({max_length}).")
    valid = sorted({int(b) for b in buckets if 0 < int(b) <= max_length})
    if not valid or valid[-1] < max_length:
        valid.append(int(max_length))
    for bucket in valid:
        if seq_len <= bucket:
            return bucket
    return int(max_length)


def static_pad_embeds(
    embeds: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    bucket_size: int | None = None,
    buckets: tuple[int, ...] | list[int] = DEFAULT_PROMPT_BUCKETS,
    max_length: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeds.ndim != 3 or embeds.shape[1] < 1:
        raise ValueError("embeds must be a nonempty [B, S, C] tensor.")
    batch, seq_len, dim = embeds.shape
    if attention_mask is None:
        mask = torch.ones((batch, seq_len), dtype=torch.bool, device=embeds.device)
    else:
        if attention_mask.shape != (batch, seq_len):
            raise ValueError(f"attention_mask must have shape {(batch, seq_len)}.")
        if attention_mask.dtype != torch.bool and not torch.all(
            (attention_mask == 0) | (attention_mask == 1)
        ):
            raise ValueError("attention_mask must be a binary [B, S] mask.")
        mask = attention_mask.ne(0)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Each sample needs at least one valid text token.")

    if bucket_size is None:
        target_len = select_prompt_bucket(seq_len, buckets=buckets, max_length=max_length)
    else:
        target_len = int(bucket_size)
        if target_len < seq_len:
            raise ValueError(f"bucket_size ({target_len}) must be >= sequence length ({seq_len}).")

    if target_len == seq_len:
        return embeds, mask

    padded_embeds = embeds.new_zeros((batch, target_len, dim))
    padded_embeds[:, :seq_len] = embeds
    padded_mask = torch.zeros((batch, target_len), dtype=torch.bool, device=embeds.device)
    padded_mask[:, :seq_len] = mask
    return padded_embeds, padded_mask


def compile_transformer(transformer, *, backend: str = TPU_BACKEND, fullgraph: bool = True):
    """Compile MOSSImageTransformer statically (dynamic=False) for persistent XLA caching."""
    transformer.compile(backend=backend, dynamic=False, fullgraph=fullgraph)
    return transformer


def _contiguous_group_norm_pre_hook(_module, args, kwargs):
    """Force NCHW-contiguous inputs so TorchTPU GroupNorm does not reject channels-last strides."""
    if args and isinstance(args[0], torch.Tensor) and not args[0].is_contiguous():
        return (args[0].contiguous(), *args[1:]), kwargs
    tensor = kwargs.get("input")
    if isinstance(tensor, torch.Tensor) and not tensor.is_contiguous():
        updated = dict(kwargs)
        updated["input"] = tensor.contiguous()
        return args, updated
    return None


def ensure_group_norm_contiguous(module: torch.nn.Module | None) -> int:
    """Attach an idempotent pre-hook ensuring contiguous tensors for every nn.GroupNorm child."""
    if module is None or not hasattr(module, "modules"):
        return 0
    hooked = 0
    for child in module.modules():
        if isinstance(child, torch.nn.GroupNorm) and not getattr(
            child, "_tpu_contiguous_group_norm_hook", False
        ):
            child.register_forward_pre_hook(_contiguous_group_norm_pre_hook, with_kwargs=True)
            child._tpu_contiguous_group_norm_hook = True
            hooked += 1
    return hooked


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(
            f"Unsupported MOSS-Image pipeline source for TPU runtime patch at {label} "
            f"(expected 1 match, found {count})."
        )
    return text.replace(old, new, 1)


_OLD_ATTENTION = """def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)"""

_NEW_ATTENTION = """def _is_compiling():
    compiler = getattr(torch, 'compiler', None)
    fn = getattr(compiler, 'is_compiling', None)
    return bool(fn()) if fn is not None else False


def _same_device(a, b):
    return a.type == b.type and (a.index or 0) == (b.index or 0)


def _attention(q, k, v, rope, heads, attn_mask=None):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    if (
        attn_mask is not None
        and attn_mask.ndim == 4
        and attn_mask.shape[-2] == 1
        and q.shape[-2] != 1
    ):
        # TorchTPU SDPA misreads or NaNs 4D masks that broadcast the query axis when S >= 512.
        attn_mask = attn_mask.expand(
            attn_mask.shape[0], attn_mask.shape[1], q.shape[-2], attn_mask.shape[-1]
        )
    return F.scaled_dot_product_attention(
        q.to(v.dtype), k.to(v.dtype), v, attn_mask=attn_mask
    ).transpose(1, 2).flatten(2)"""

_OLD_TEXT_BLOCK_FORWARD = """    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))"""

_NEW_TEXT_BLOCK_FORWARD = """    def forward(self, x, rope, attn_mask=None):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads, attn_mask=attn_mask))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))"""

_OLD_TEXT_PROJECTION_FORWARD = """    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x"""

_NEW_TEXT_PROJECTION_FORWARD = """    def forward(self, x, rope, attn_mask=None):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope, attn_mask=attn_mask)
        return x"""

_OLD_BLOCK_FORWARD = """    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj['q'].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj['o'](_attention(q, k, self.proj['v'](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj['ffn_up_gate'](y.to(dtype)).chunk(2, -1)
        return x + self.proj['ffn_down'](_situ(hidden, gate)) * gate_f[:, None]"""

_NEW_BLOCK_FORWARD = """    def forward(self, x, rope, modulation, attn_mask=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj['q'].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj['o'](_attention(q, k, self.proj['v'](y), rope, self.heads, attn_mask=attn_mask)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj['ffn_up_gate'](y.to(dtype)).chunk(2, -1)
        return x + self.proj['ffn_down'](_situ(hidden, gate)) * gate_f[:, None]"""

_OLD_ROPE_FORWARD = """    def forward(self, height, width, text_length, device):
        # Diffusers/Accelerate can install tensors directly, bypassing load hooks.
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        # Coordinates: (r, f, h, w); sequence order is image followed by text.
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)"""

_NEW_ROPE_FORWARD = """    def forward(self, height, width, text_length, device, text_positions=None, text_offset=None):
        # Diffusers/Accelerate can install tensors directly, bypassing load hooks.
        if self.bands.dtype != torch.float32 or not _same_device(self.bands.device, device):
            self.bands = self._default_bands(device)
        # Coordinates: (r, f, h, w); sequence order is image followed by text.
        h, w = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing='ij',
        )
        zeros_hw = torch.zeros_like(h)
        if text_offset is None:
            f_img = torch.full_like(h, text_length)
        else:
            f_img = zeros_hw + text_offset.to(dtype=torch.float32, device=device)
        image = torch.stack(
            (zeros_hw, f_img, h - (height - 1) / 2, w - (width - 1) / 2), -1
        ).flatten(0, 1)
        if text_positions is None:
            pos = torch.arange(text_length, device=device, dtype=torch.float32)
        else:
            pos = text_positions.to(dtype=torch.float32, device=device)
        zeros_t = torch.zeros_like(pos)
        text = torch.stack((zeros_t, pos, zeros_t, zeros_t), -1)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)"""

_OLD_TRANSFORMER_INIT_TAIL = """        self.video_head_modulation = nn.Linear(dim, 2 * dim)"""

_NEW_TRANSFORMER_INIT_TAIL = """        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self.register_buffer(
            '_tpu_time_freq',
            self._build_tpu_time_freq(freq_dim // 2, torch.device('cpu')),
            persistent=False,
        )

    @staticmethod
    def _build_tpu_time_freq(half, device):
        exponent = -torch.arange(half, device='cpu', dtype=torch.float64) / half
        return (10000.0 ** exponent).to(dtype=torch.float32, device=device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        freq = getattr(self, '_tpu_time_freq', None)
        if freq is not None and not freq.is_meta and freq.dtype != torch.float32:
            self._tpu_time_freq = self._build_tpu_time_freq(self.config.freq_dim // 2, freq.device)
        return self

    def _get_tpu_time_freq(self, device):
        freq = getattr(self, '_tpu_time_freq', None)
        if (
            freq is None
            or freq.is_meta
            or freq.dtype != torch.float32
            or not _same_device(freq.device, device)
        ):
            freq = self._build_tpu_time_freq(self.config.freq_dim // 2, device)
            if not _is_compiling():
                self._tpu_time_freq = freq
        return freq"""

_OLD_TRANSFORMER_FORWARD = """    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=None, return_dict=True, skip_layers=None):
        skip_layers = _validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
            raise ValueError('hidden_states must be [B, latent_channels, H, W].')
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError('encoder_hidden_states must be [B, S, text_dim].')
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('encoder_attention_mask must be a binary [B, S] mask.')
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        # Training computes the sinusoid in float64 and feeds float32 into its MLP.
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index:index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError('Each sample needs at least one valid text token.')
            video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
            # Match the reference activation dtype under fp32 weights + autocast.
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width:])
            x = torch.cat((video, text), 1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    x = layer(x, rope, mod[index:index + 1])
            x = self.final_norm_video(x[:, :height * width].to(self.final_norm_video.weight.dtype))
            x = x * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        sample = torch.cat(outputs)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)"""

_NEW_TRANSFORMER_FORWARD = """    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=None, return_dict=True, skip_layers=None):
        skip_layers = _validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
            raise ValueError('hidden_states must be [B, latent_channels, H, W].')
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError('encoder_hidden_states must be [B, S, text_dim].')
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2]:
            raise ValueError('encoder_attention_mask must be a binary [B, S] mask.')
        is_compiling = _is_compiling()
        validate_mask = (
            not is_compiling
            and not getattr(self, '_skip_runtime_mask_checks', False)
            and hidden_states.device.type not in ('tpu', 'mps')
        )
        if validate_mask and mask.dtype != torch.bool and not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('encoder_attention_mask must be a binary [B, S] mask.')
        times = (
            timestep.to(device=hidden_states.device, dtype=torch.float32)
            if isinstance(timestep, torch.Tensor)
            else torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32)
        ).reshape(-1)
        times = times.expand(batch)
        # TPU MXU runs float32 natively without float64 fallback or host sync.
        if times.device.type in ('tpu', 'mps') or getattr(self, '_force_tpu_timestep_sinusoid', False):
            freq = self._get_tpu_time_freq(times.device)
            phase = times[:, None] * freq
        else:
            half = self.config.freq_dim // 2
            freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
            phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        use_static_mask = (
            is_compiling
            or hidden_states.device.type == 'tpu'
            or getattr(self, '_static_attention_masking', False)
        )
        outputs = []
        image_valid = None
        if use_static_mask:
            image_valid = torch.ones((1, 1, 1, height * width), dtype=torch.bool, device=hidden_states.device)
        for index in range(batch):
            video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
            if use_static_mask:
                valid_mask = mask[index:index + 1].ne(0)
                if validate_mask and not torch.any(valid_mask):
                    raise ValueError('Each sample needs at least one valid text token.')
                zero = encoder_hidden_states.new_zeros(())
                context = torch.where(valid_mask.unsqueeze(-1), encoder_hidden_states[index:index + 1], zero)
                mask_i32 = valid_mask[0].to(torch.int32)
                true_len = mask_i32.sum().to(torch.float32)
                text_pos = (mask_i32.cumsum(0) - 1).clamp(min=0).to(torch.float32)
                rope = self.rope(
                    height,
                    width,
                    encoder_hidden_states.shape[1],
                    hidden_states.device,
                    text_positions=text_pos,
                    text_offset=true_len,
                ).to(video.dtype)
                text_len = encoder_hidden_states.shape[1]
                joint_len = height * width + text_len
                text_mask_1d = valid_mask[:, None, None, :]
                text_attn_mask = text_mask_1d.expand(1, 1, text_len, text_len)
                joint_attn_mask = torch.cat((image_valid, text_mask_1d), dim=-1).expand(
                    1, 1, joint_len, joint_len
                )
                text = self.text_embedder(context, rope[:, height * width:], attn_mask=text_attn_mask)
            else:
                context = encoder_hidden_states[index:index + 1, mask[index].bool()]
                if context.shape[1] == 0:
                    raise ValueError('Each sample needs at least one valid text token.')
                # Match the reference activation dtype under fp32 weights + autocast.
                rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
                joint_attn_mask = None
                text = self.text_embedder(context, rope[:, height * width:])
            x = torch.cat((video, text), 1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    x = layer(x, rope, mod[index:index + 1], attn_mask=joint_attn_mask)
            x = self.final_norm_video(x[:, :height * width].to(self.final_norm_video.weight.dtype))
            x = x * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        sample = torch.cat(outputs)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)"""


_PATCHES = (
    ("_attention", _OLD_ATTENTION, _NEW_ATTENTION),
    ("_TextBlock.forward", _OLD_TEXT_BLOCK_FORWARD, _NEW_TEXT_BLOCK_FORWARD),
    ("_TextProjection.forward", _OLD_TEXT_PROJECTION_FORWARD, _NEW_TEXT_PROJECTION_FORWARD),
    ("_Block.forward", _OLD_BLOCK_FORWARD, _NEW_BLOCK_FORWARD),
    ("_RoPE.forward", _OLD_ROPE_FORWARD, _NEW_ROPE_FORWARD),
    ("MOSSImageTransformer.__init__", _OLD_TRANSFORMER_INIT_TAIL, _NEW_TRANSFORMER_INIT_TAIL),
    ("MOSSImageTransformer.forward", _OLD_TRANSFORMER_FORWARD, _NEW_TRANSFORMER_FORWARD),
    (
        "MOSSImagePipeline",
        "class MOSSImagePipeline(DiffusionPipeline):",
        """class MOSSImagePipeline(DiffusionPipeline):
    @property
    def _execution_device(self):
        device = getattr(self, '_tpu_execution_device', None)
        return device if device is not None else super()._execution_device
""",
    ),
)


def _patch_pipeline_source(text: str) -> str:
    patched = text
    for label, old, new in _PATCHES:
        patched = _replace_once(patched, old, new, label)
    return patched


def _import_pipeline_module(root: Path):
    pipeline_file = (Path(root) / "pipeline_moss_image.py").resolve()
    raw_source = pipeline_file.read_text(encoding="utf-8")
    source_digest = hashlib.sha256(raw_source.encode("utf-8")).hexdigest()
    cache_key = str(pipeline_file)
    cached = _PATCHED_MODULES.get(cache_key)
    if cached is not None and cached[0] == source_digest:
        return cached[1], pipeline_file

    patched = _patch_pipeline_source(raw_source)
    module_name = f"moss_image_tpu_pipeline_{source_digest[:12]}"
    spec = importlib.util.spec_from_loader(module_name, loader=None, origin=str(pipeline_file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    exec(compile(patched, str(pipeline_file), "exec"), module.__dict__)
    _PATCHED_MODULES[cache_key] = (source_digest, module)
    return module, pipeline_file


def _validate_binary_mask_tensor(mask: torch.Tensor | None) -> None:
    if mask is not None and mask.dtype != torch.bool and not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
    if mask is not None and not torch.all(mask.ne(0).any(dim=1)):
        raise ValueError("Each sample needs at least one valid text token.")


def _round_opt(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


class StagedTPUImagePipeline:
    """Staged / compiled Cloud TPU pipeline preserving Qwen3.5-9B fidelity and static DiT buckets."""

    def __init__(
        self,
        pipe,
        pipeline_file: Path,
        *,
        device: torch.device | None = None,
        offload: str = "model",
        stage_switch_mode: str = "auto",
        prompt_buckets: tuple[int, ...] = DEFAULT_PROMPT_BUCKETS,
        compile_transformer_enabled: bool = False,
        eager_mode: str = "default",
        text_eager_mode: str | None = None,
        matmul_precision: str = "high",
        prompt_cache_dir: str | Path | None = None,
        prompt_cache_model_id: str | None = None,
        gdn_pallas: bool = False,
        gdn_buckets: tuple[int, ...] | None = tpu_qwen35.DEFAULT_GDN_BUCKETS,
    ):
        self._base_pipe = pipe
        self.pipeline_file = Path(pipeline_file).resolve()
        self.wrapper_file = Path(__file__).resolve()
        self._runtime_device = torch.device(device) if device is not None else tpu_device()
        self._base_pipe._tpu_execution_device = self._runtime_device
        self._cpu_device = torch.device("cpu")
        self._prompt_buckets = tuple(int(b) for b in prompt_buckets)
        self._eager_mode = validate_eager_mode(eager_mode)
        self._text_eager_mode = (
            self._eager_mode
            if text_eager_mode is None
            else validate_eager_mode(text_eager_mode, label="TPU text eager mode")
        )
        self._stage_switch_mode = validate_stage_switch_mode(stage_switch_mode)
        self._force_drop_tpu_for_test = False
        self._cpu_weight_snapshots: dict[str, dict[str, list]] = {}
        self._meta_offloaded_modules: set[str] = set()
        self._tpu_resident_modules: set[str] = set()
        self._stage_switch_history: list[dict[str, object]] = []
        self._matmul_precision = configure_precision(matmul_precision)
        self._prompt_cache: OrderedDict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._prompt_cache_dir = Path(prompt_cache_dir) if prompt_cache_dir is not None else None
        self._prompt_cache_model_id = str(prompt_cache_model_id or self.pipeline_file.parent.resolve())
        self._gdn_patch_handle = None
        self._stage = "cpu"
        self._compiled = False
        self._lock = RLock()
        ensure_group_norm_contiguous(self.vae)
        self.configure_offload(offload)

        if gdn_pallas and self.text_encoder is not None:
            self.enable_gdn_pallas(buckets=gdn_buckets)

        if compile_transformer_enabled:
            self.enable_transformer_compile()

    @property
    def eager_mode(self) -> str:
        return self._eager_mode

    @property
    def text_eager_mode(self) -> str:
        return self._text_eager_mode

    @property
    def stage_switch_mode(self) -> str:
        return self._stage_switch_mode

    @property
    def effective_stage_switch_mode(self) -> str:
        return resolve_stage_switch_mode(
            self._stage_switch_mode,
            self._runtime_device,
            force_drop_tpu=self._force_drop_tpu_for_test,
        )

    @property
    def stage_switch_history(self) -> list[dict[str, object]]:
        return list(self._stage_switch_history)

    def reset_stage_switch_history(self) -> None:
        with self._lock:
            self._stage_switch_history.clear()

    def set_stage_switch_mode(self, mode: str, *, clear_snapshots: bool = False) -> str:
        validated = validate_stage_switch_mode(mode)
        with self._lock:
            self._stage_switch_mode = validated
            new_effective = self.effective_stage_switch_mode
            if new_effective == "d2h" and (clear_snapshots or validated == "d2h"):
                module_map = {
                    "text_encoder": self.text_encoder,
                    "transformer": self.transformer,
                    "vae": self.vae,
                }
                for name in list(self._meta_offloaded_modules):
                    mod = module_map.get(name)
                    snap = self._cpu_weight_snapshots.get(name)
                    if mod is not None and snap is not None:
                        _restore_module_from_cpu_snapshot(mod, snap, self._cpu_device)
                    self._meta_offloaded_modules.discard(name)
                self._cpu_weight_snapshots.clear()
        return self._stage_switch_mode

    @property
    def transformer(self):
        return self._base_pipe.transformer

    @transformer.setter
    def transformer(self, value):
        self._base_pipe.transformer = value
        self._cpu_weight_snapshots.pop("transformer", None)
        self._meta_offloaded_modules.discard("transformer")
        self._tpu_resident_modules.discard("transformer")

    @property
    def text_encoder(self):
        return self._base_pipe.text_encoder

    @text_encoder.setter
    def text_encoder(self, value):
        self._base_pipe.text_encoder = value
        self._prompt_cache.clear()
        self._cpu_weight_snapshots.pop("text_encoder", None)
        self._meta_offloaded_modules.discard("text_encoder")
        self._tpu_resident_modules.discard("text_encoder")

    @property
    def vae(self):
        return self._base_pipe.vae

    @vae.setter
    def vae(self, value):
        self._base_pipe.vae = value
        self._cpu_weight_snapshots.pop("vae", None)
        self._meta_offloaded_modules.discard("vae")
        self._tpu_resident_modules.discard("vae")
        ensure_group_norm_contiguous(value)

    @property
    def tokenizer(self):
        return self._base_pipe.tokenizer

    @property
    def scheduler(self):
        return self._base_pipe.scheduler

    @scheduler.setter
    def scheduler(self, value):
        self._base_pipe.scheduler = value

    @property
    def progress_bar(self):
        return self._base_pipe.progress_bar

    @progress_bar.setter
    def progress_bar(self, value):
        self._base_pipe.progress_bar = value

    @property
    def components(self):
        return self._base_pipe.components

    def set_progress_bar_config(self, **kwargs):
        return self._base_pipe.set_progress_bar_config(**kwargs)

    def register_to_config(self, **kwargs):
        return self._base_pipe.register_to_config(**kwargs)

    def remove_all_hooks(self):
        fn = getattr(self._base_pipe, "remove_all_hooks", None)
        if callable(fn):
            return fn()
        return None

    def enable_gdn_pallas(
        self,
        *,
        buckets: tuple[int, ...] | None = tpu_qwen35.DEFAULT_GDN_BUCKETS,
        modeling_module=None,
        op_resolver=None,
        validate_op: bool = True,
        require_tpu_device: bool = True,
    ):
        """Enable experimental deltanet-tpu Pallas GDN kernel patching on text_encoder."""
        self.disable_gdn_pallas()
        self._gdn_patch_handle = tpu_qwen35.install_qwen35_gdn_pallas_patch(
            model=self.text_encoder,
            modeling_module=modeling_module,
            buckets=buckets,
            op_resolver=op_resolver,
            validate_op=validate_op,
            require_tpu_device=require_tpu_device,
        )
        return self._gdn_patch_handle

    def disable_gdn_pallas(self) -> None:
        """Restore the original Qwen3.5 GDN implementation if patched by this pipeline."""
        if self._gdn_patch_handle is not None:
            self._gdn_patch_handle.restore()
            self._gdn_patch_handle = None

    @property
    def gdn_pallas_stats(self) -> dict | None:
        if self._gdn_patch_handle is None:
            return None
        stats = self._gdn_patch_handle.stats
        return {
            "pallas_calls": int(stats["pallas_calls"]),
            "fallback_calls": int(stats["fallback_calls"]),
            "fallback_reasons": dict(stats["fallback_reasons"]),
        }

    def configure_offload(self, offload: str):
        if offload not in VALID_OFFLOAD_MODES:
            raise ValueError(f"Unknown TPU offload mode: {offload}. Expected one of {VALID_OFFLOAD_MODES}.")
        if offload == "auto":
            offload = "none" if tpu_budget_gib() >= 40 else "model"
        offload = "model" if offload == "staged" else offload
        with self._lock, configure_eager_mode(self._eager_mode, self._runtime_device):
            self._offload = offload
            if offload == "none":
                self._activate_stage("all")
            elif self._stage == "all":
                self._activate_stage("diffusion")

    def enable_transformer_compile(self, backend: str = TPU_BACKEND, fullgraph: bool = True):
        if self._compiled:
            return self.transformer
        compile_transformer(self.transformer, backend=backend, fullgraph=fullgraph)
        self._compiled = True
        return self.transformer

    def _activate_stage(self, stage: str) -> None:
        effective_mode = self.effective_stage_switch_mode
        retain_snapshots = effective_mode == "drop_tpu" and stage != "all"
        modules = (
            ("text_encoder", self.text_encoder),
            ("transformer", self.transformer),
            ("vae", self.vae),
        )
        available = {name for name, module in modules if module is not None}
        if stage == "cpu":
            desired: set[str] = set()
        elif stage == "all":
            desired = set(available)
        elif stage == "diffusion":
            desired = available & {"transformer", "vae"}
        elif stage == "text":
            desired = available & {"text_encoder"}
            # Keep the lightweight VAE (~0.3 GiB) resident once loaded across diffusion <-> text switches,
            # while avoiding unnecessary VAE loading on cold cpu -> text runs (e.g. encode-only).
            if "vae" in self._tpu_resident_modules and "vae" in available:
                desired.add("vae")
        else:
            raise ValueError(f"Unknown TPU stage: {stage!r}.")

        offload_list = [
            (name, module)
            for name, module in modules
            if module is not None and name in self._tpu_resident_modules and name not in desired
        ]
        load_list = [
            (name, module)
            for name, module in modules
            if module is not None and name in desired and name not in self._tpu_resident_modules
        ]
        if self._stage == stage and not offload_list and not load_list:
            return
        host_before = host_memory_info_gib()
        tpu_before = memory_allocated_gib()
        t_total0 = _perf_counter()
        pre_sync_seconds = 0.0
        d2h_seconds = 0.0
        drop_tpu_seconds = 0.0
        gc_seconds = 0.0
        empty_cache_seconds = 0.0
        h2d_seconds = 0.0
        tpu_after_offload = tpu_before

        try:
            if self._stage != "cpu" and (offload_list or load_list):
                t_pre0 = _perf_counter()
                synchronize()
                pre_sync_seconds = _perf_counter() - t_pre0

            if offload_list:
                t_off0 = _perf_counter()
                if effective_mode == "drop_tpu":
                    for name, module in offload_list:
                        if name not in self._cpu_weight_snapshots:
                            self._cpu_weight_snapshots[name] = _snapshot_module_cpu_weights(module)
                        _release_module_to_meta(module)
                        self._meta_offloaded_modules.add(name)
                    drop_tpu_seconds = _perf_counter() - t_off0
                    t_gc0 = _perf_counter()
                    gc.collect()
                    gc_seconds = _perf_counter() - t_gc0
                    self._tpu_resident_modules.difference_update(name for name, _ in offload_list)
                else:
                    for name, module in offload_list:
                        if name in self._meta_offloaded_modules and name in self._cpu_weight_snapshots:
                            _restore_module_from_cpu_snapshot(
                                module, self._cpu_weight_snapshots[name], self._cpu_device
                            )
                            self._meta_offloaded_modules.discard(name)
                        else:
                            module.to(self._cpu_device)
                    synchronize()
                    d2h_seconds = _perf_counter() - t_off0
                    self._tpu_resident_modules.difference_update(name for name, _ in offload_list)

            if self._stage != "cpu":
                t_ec0 = _perf_counter()
                empty_cache()
                empty_cache_seconds = _perf_counter() - t_ec0
                tpu_after_offload = memory_allocated_gib()

            if load_list:
                t_load0 = _perf_counter()
                use_non_blocking = self._runtime_device.type != "cpu"
                pending_sources: list[torch.Tensor] = []
                for name, module in load_list:
                    if retain_snapshots and name not in self._cpu_weight_snapshots:
                        self._cpu_weight_snapshots[name] = _snapshot_module_cpu_weights(module)
                    if name in self._meta_offloaded_modules and name in self._cpu_weight_snapshots:
                        _restore_module_from_cpu_snapshot(
                            module, self._cpu_weight_snapshots[name], self._runtime_device
                        )
                        self._meta_offloaded_modules.discard(name)
                    else:
                        if use_non_blocking:
                            # Keep source host tensors alive until synchronize() finishes async DMA.
                            pending_sources.extend(
                                p.data for p in module.parameters() if not p.data.is_meta
                            )
                            pending_sources.extend(
                                b for b in module.buffers() if not b.is_meta
                            )
                        module.to(self._runtime_device, non_blocking=use_non_blocking)
                synchronize()
                pending_sources.clear()
                h2d_seconds = _perf_counter() - t_load0
                self._tpu_resident_modules.update(name for name, _ in load_list)

            if stage == "all":
                self._cpu_weight_snapshots.clear()
                self._meta_offloaded_modules.clear()
        except Exception:
            for name, module in modules:
                if module is None:
                    continue
                snap = self._cpu_weight_snapshots.get(name)
                if effective_mode == "drop_tpu" and snap is not None:
                    _release_module_to_meta(module)
                    self._meta_offloaded_modules.add(name)
                elif name in self._meta_offloaded_modules and snap is not None:
                    _restore_module_from_cpu_snapshot(module, snap, self._cpu_device)
                    self._meta_offloaded_modules.discard(name)
                else:
                    module.to(self._cpu_device)
            self._tpu_resident_modules.clear()
            self._stage = "cpu"
            empty_cache()
            raise

        total_seconds = _perf_counter() - t_total0
        tpu_after_load = memory_allocated_gib()
        host_after = host_memory_info_gib()
        self._stage_switch_history.append(
            {
                "from_stage": self._stage,
                "to_stage": stage,
                "switch_mode": effective_mode,
                "offload_modules": [name for name, _ in offload_list],
                "load_modules": [name for name, _ in load_list],
                "pre_sync_seconds": round(pre_sync_seconds, 4),
                "d2h_seconds": round(d2h_seconds, 4),
                "drop_tpu_seconds": round(drop_tpu_seconds, 4),
                "gc_seconds": round(gc_seconds, 4),
                "empty_cache_seconds": round(empty_cache_seconds, 4),
                "h2d_seconds": round(h2d_seconds, 4),
                "total_seconds": round(total_seconds, 4),
                "tpu_allocated_before_gib": _round_opt(tpu_before, 3),
                "tpu_allocated_after_offload_gib": _round_opt(tpu_after_offload, 3),
                "tpu_allocated_after_load_gib": _round_opt(tpu_after_load, 3),
                "host_mem_before": host_before,
                "host_mem_after": host_after,
            }
        )
        self._stage = stage

    def _get_cached_or_encode(
        self, prompts: list[str], max_sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not prompts or not all(isinstance(p, str) for p in prompts):
            raise ValueError("prompt must be a string or a nonempty list of strings.")

        keys = dict.fromkeys((prompt, max_sequence_length) for prompt in prompts)
        pairs = {key: self._prompt_cache[key] for key in keys if key in self._prompt_cache}
        missing = keys.keys() - pairs.keys()
        if missing and self._prompt_cache_dir is not None:
            for key in list(missing):
                cached_disk = tpu_qwen35.load_cached_prompt_embed(
                    self._prompt_cache_dir, self._prompt_cache_model_id, key[0], key[1]
                )
                if cached_disk is not None:
                    pairs[key] = cached_disk
            missing = keys.keys() - pairs.keys()
        if missing:
            if self.text_encoder is None or self.tokenizer is None:
                raise ValueError("Text encoder/tokenizer required for string prompts.")
            self._activate_stage("text" if self._offload == "model" else "all")
            for key in keys:
                if key in missing:
                    embeds, mask = self._base_pipe.encode_prompt(key[0], max_sequence_length=max_sequence_length)
                    pair = (embeds[0].detach().cpu(), mask[0].detach().cpu().to(torch.bool))
                    pairs[key] = pair
                    if self._prompt_cache_dir is not None:
                        tpu_qwen35.save_cached_prompt_embed(
                            self._prompt_cache_dir,
                            self._prompt_cache_model_id,
                            key[0],
                            key[1],
                            pair[0],
                            pair[1],
                        )

        rows = []
        masks = []
        for text_item in prompts:
            key = (text_item, max_sequence_length)
            emb, msk = pairs[key]
            self._prompt_cache[key] = (emb, msk)
            self._prompt_cache.move_to_end(key)
            rows.append(emb)
            masks.append(msk)
        while len(self._prompt_cache) > CACHED_PROMPTS:
            self._prompt_cache.popitem(last=False)

        max_len = max(r.shape[0] for r in rows)
        embeds = rows[0].new_zeros((len(rows), max_len, rows[0].shape[-1]))
        mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=embeds.device)
        for idx, (emb, msk) in enumerate(zip(rows, masks)):
            embeds[idx, : emb.shape[0]] = emb
            mask[idx, : msk.shape[0]] = msk
        return embeds, mask

    @torch.no_grad()
    def encode_prompt(self, prompt, max_sequence_length: int = 4096):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        with self._lock, configure_eager_mode(self._text_eager_mode, self._runtime_device):
            try:
                return self._get_cached_or_encode(prompts, max_sequence_length)
            finally:
                if self._stage == "text":
                    self._activate_stage("cpu")

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        *,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.0,
        negative_prompt="",
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        max_sequence_length=4096,
        output_type="pil",
        return_dict=True,
        uncond_skip_layers=(22,),
        slg_layers=None,
        slg_scale=0.0,
        slg_start=0.0,
        slg_stop=1.0,
    ):
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("height and width must be positive multiples of 16.")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be pil, np, pt or latent.")
        if num_images_per_prompt < 1 or guidance_scale < 0:
            raise ValueError("Invalid image count or guidance scale.")
        if not 1 <= max_sequence_length <= 4096:
            raise ValueError("max_sequence_length must be between 1 and 4096.")
        if (prompt is None) == (prompt_embeds is None):
            raise ValueError("Provide exactly one of prompt and prompt_embeds.")

        with self._lock:
            try:
                do_cfg = guidance_scale != 1.0
                if prompt_embeds is None:
                    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
                    batch = len(prompts)
                else:
                    _validate_binary_mask_tensor(prompt_attention_mask)
                    batch = prompt_embeds.shape[0]
                    prompts = None

                negatives = None
                if do_cfg:
                    if negative_prompt_embeds is None:
                        negatives = (
                            [negative_prompt] * batch
                            if isinstance(negative_prompt, str)
                            else list(negative_prompt)
                        )
                        if len(negatives) != batch:
                            raise ValueError("negative_prompt must match prompt batch size.")
                    else:
                        _validate_binary_mask_tensor(negative_prompt_attention_mask)

                if prompts is not None or negatives is not None:
                    with configure_eager_mode(self._text_eager_mode, self._runtime_device):
                        if prompts is not None:
                            prompt_embeds, prompt_attention_mask = self._get_cached_or_encode(
                                prompts, max_sequence_length
                            )
                        if negatives is not None:
                            negative_prompt_embeds, negative_prompt_attention_mask = (
                                self._get_cached_or_encode(negatives, max_sequence_length)
                            )

                with configure_eager_mode(self._eager_mode, self._runtime_device):
                    self._activate_stage("diffusion" if self._offload == "model" else "all")

                    pos_embeds, pos_mask = static_pad_embeds(
                        prompt_embeds,
                        prompt_attention_mask,
                        buckets=self._prompt_buckets,
                        max_length=max_sequence_length,
                    )
                    neg_embeds, neg_mask = (None, None)
                    if do_cfg:
                        neg_embeds, neg_mask = static_pad_embeds(
                            negative_prompt_embeds,
                            negative_prompt_attention_mask,
                            buckets=self._prompt_buckets,
                            max_length=max_sequence_length,
                        )

                    return self._base_pipe(
                        prompt=None,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        negative_prompt=negative_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        generator=generator,
                        latents=latents,
                        prompt_embeds=pos_embeds,
                        prompt_attention_mask=pos_mask,
                        negative_prompt_embeds=neg_embeds,
                        negative_prompt_attention_mask=neg_mask,
                        max_sequence_length=max_sequence_length,
                        output_type=output_type,
                        return_dict=return_dict,
                        uncond_skip_layers=uncond_skip_layers,
                        slg_layers=slg_layers,
                        slg_scale=slg_scale,
                        slg_start=slg_start,
                        slg_stop=slg_stop,
                    )
            finally:
                if self._stage == "text":
                    self._activate_stage("cpu")


def load_pipeline(
    model: str,
    *,
    offload: str = "auto",
    stage_switch_mode: str = "auto",
    compile_transformer_enabled: bool = False,
    compile_transformer: bool | None = None,
    local_files_only: bool = False,
    prompt_buckets: tuple[int, ...] = DEFAULT_PROMPT_BUCKETS,
    eager_mode: str = "default",
    text_eager_mode: str | None = None,
    matmul_precision: str = "high",
    include_text_encoder: bool = True,
    device: torch.device | None = None,
    prompt_cache_dir: str | Path | None = None,
    gdn_pallas: bool = False,
    gdn_buckets: tuple[int, ...] | None = tpu_qwen35.DEFAULT_GDN_BUCKETS,
):
    if compile_transformer is not None:
        compile_transformer_enabled = bool(compile_transformer)

    components = TEXT_COMPONENTS + DIFFUSION_COMPONENTS if include_text_encoder else DIFFUSION_COMPONENTS
    root = resolve_model_bundle(model, local_files_only=local_files_only, components=components)
    module, pipeline_file = _import_pipeline_module(root)

    common = {"local_files_only": local_files_only}
    transformer = module.MOSSImageTransformer.from_pretrained(
        Path(root) / "transformer",
        torch_dtype=torch.bfloat16,
        **common,
    ).eval()

    text_encoder = None
    tokenizer = None
    if include_text_encoder:
        root_path = Path(root)
        tokenizer_path = root_path / "tokenizer"
        if not tokenizer_path.is_dir():
            tokenizer_path = root_path / "text_encoder"
        tokenizer = module.AutoTokenizer.from_pretrained(tokenizer_path, **common)
        text_encoder = module.AutoModel.from_pretrained(
            root_path / "text_encoder",
            dtype=torch.bfloat16,
            **common,
        ).eval()

    vae_root = Path(root) / "vae"
    subfolder = None if (vae_root / "config.json").is_file() else "vae"
    vae = module.AutoencoderKLFlux2.from_pretrained(
        vae_root,
        subfolder=subfolder,
        torch_dtype=torch.float32,
        **common,
    ).eval()

    scheduler_dir = Path(root) / "scheduler"
    if scheduler_dir.is_dir():
        scheduler = module.MOSSImageScheduler.from_pretrained(scheduler_dir, **common)
    else:
        scheduler = module.MOSSImageScheduler()

    base_pipe = module.MOSSImagePipeline(
        transformer=transformer,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
    )
    return StagedTPUImagePipeline(
        base_pipe,
        pipeline_file=pipeline_file,
        device=device,
        offload=offload,
        stage_switch_mode=stage_switch_mode,
        prompt_buckets=prompt_buckets,
        compile_transformer_enabled=compile_transformer_enabled,
        eager_mode=eager_mode,
        text_eager_mode=text_eager_mode,
        matmul_precision=matmul_precision,
        prompt_cache_dir=prompt_cache_dir,
        prompt_cache_model_id=model,
        gdn_pallas=gdn_pallas,
        gdn_buckets=gdn_buckets,
    )
