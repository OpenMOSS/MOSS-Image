"""Experimental Qwen3.5-9B Stage-1 TPU helpers (Pallas GDN bridge + persistent prompt cache).

Do NOT add ``from __future__ import annotations`` to this module: ``torch.tpu.pallas.jax_op``
inspects wrapped kernel parameter annotations at runtime and expects concrete ``jax.Array``
types rather than PEP-563 string annotations.
"""

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Callable

import torch
import torch.nn.functional as F

GDN_CHUNK_SIZE = 128
DEFAULT_GDN_BUCKETS = (128, 256, 384, 512, 768, 1024)
PROMPT_CACHE_VERSION = 1

_LOCK = RLock()
_JAX_BACKEND_LOCK = RLock()
_GDN_OP_CACHE: dict[tuple[bool, str], Any] = {}


def is_deltanet_tpu_available() -> bool:
    """Return True when JAX, deltanet-tpu, and torch.tpu.pallas.jax_op are importable."""
    if importlib.util.find_spec("jax") is None or importlib.util.find_spec("deltanet_tpu") is None:
        return False
    tpu_mod = getattr(torch, "tpu", None)
    pallas_mod = getattr(tpu_mod, "pallas", None)
    return callable(getattr(pallas_mod, "jax_op", None))


def normalize_gdn_buckets(buckets: str | tuple[int, ...] | list[int] | None) -> tuple[int, ...] | None:
    """Validate and sort static sequence buckets (each must be a positive multiple of 128)."""
    if buckets is None:
        return None
    if isinstance(buckets, str):
        values = [int(part.strip()) for part in buckets.split(",") if part.strip()]
    else:
        values = [int(v) for v in buckets]
    if not values or any(v <= 0 or (v % GDN_CHUNK_SIZE) != 0 for v in values):
        raise ValueError(
            f"GDN buckets must be positive multiples of {GDN_CHUNK_SIZE}, got {buckets!r}."
        )
    return tuple(sorted(set(values)))


def select_gdn_bucket(seq_len: int, buckets: tuple[int, ...] | None = DEFAULT_GDN_BUCKETS) -> int:
    """Return a chunk-aligned static bucket length >= seq_len for causal GDN execution."""
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}.")
    if buckets is None:
        return ((seq_len + GDN_CHUNK_SIZE - 1) // GDN_CHUNK_SIZE) * GDN_CHUNK_SIZE
    for bucket in buckets:
        if bucket >= seq_len:
            return bucket
    return ((seq_len + GDN_CHUNK_SIZE - 1) // GDN_CHUNK_SIZE) * GDN_CHUNK_SIZE


@contextmanager
def _force_tpu_jax_backend(jax_module: Any):
    """Temporarily report jax.default_backend() == 'tpu' while tracing Pallas via jax_op."""
    interpret_flag = os.environ.get("PALLAS_INTERPRET", "").strip().lower()
    if interpret_flag in ("1", "true", "yes"):
        yield
        return
    with _JAX_BACKEND_LOCK:
        orig_default_backend = jax_module.default_backend
        jax_module.default_backend = lambda: "tpu"
        try:
            yield
        finally:
            jax_module.default_backend = orig_default_backend


def register_chunk_gdn_pallas_op(
    *,
    qk_l2norm: bool = True,
    precision: str = "bf16",
    jax_op_fn: Callable[[str, Callable[..., Any]], Any] | None = None,
    jax_module: Any = None,
    chunk_gdn_fn: Callable[..., Any] | None = None,
):
    """Lazily register a pure-tensor torch.tpu.pallas.jax_op wrapping deltanet_tpu.chunk_gdn."""
    cache_key = (bool(qk_l2norm), str(precision))
    use_cache = jax_op_fn is None and jax_module is None and chunk_gdn_fn is None
    if use_cache:
        with _LOCK:
            cached = _GDN_OP_CACHE.get(cache_key)
            if cached is not None:
                return cached

    if jax_module is None:
        try:
            jax_module = importlib.import_module("jax")
        except ImportError as exc:
            raise RuntimeError(
                "JAX is required for Qwen3.5 GDN Pallas acceleration. "
                "On TorchTPU Colab, install deltanet-tpu without upgrading JAX/libtpu: "
                "`pip install --no-deps deltanet-tpu`."
            ) from exc

    if chunk_gdn_fn is None:
        try:
            deltanet_mod = importlib.import_module("deltanet_tpu")
            chunk_gdn_fn = deltanet_mod.chunk_gdn
        except ImportError as exc:
            raise RuntimeError(
                "`deltanet_tpu` is required for `--gdn-pallas`. "
                "Install with `pip install --no-deps deltanet-tpu` "
                "(do NOT install `deltanet-tpu[tpu]`, which overwrites TorchTPU's pinned libtpu)."
            ) from exc

    if jax_op_fn is None:
        tpu_mod = getattr(torch, "tpu", None)
        pallas_mod = getattr(tpu_mod, "pallas", None)
        jax_op_fn = getattr(pallas_mod, "jax_op", None)
        if not callable(jax_op_fn):
            raise RuntimeError(
                "torch.tpu.pallas.jax_op is not available in this PyTorch build."
            )

    jax_array_type = jax_module.Array

    def _chunk_gdn_kernel(
        q: jax_array_type,
        k: jax_array_type,
        v: jax_array_type,
        g: jax_array_type,
        beta: jax_array_type,
    ) -> jax_array_type:
        with _force_tpu_jax_backend(jax_module):
            res = chunk_gdn_fn(
                q,
                k,
                v,
                g,
                beta,
                chunk_size=GDN_CHUNK_SIZE,
                precision=str(precision),
                qk_l2norm=bool(qk_l2norm),
                output_final_state=False,
                is_inference=True,
            )
        out = res[0] if isinstance(res, tuple) else res
        return out.astype(v.dtype)

    op_name = f"pallas::qwen35_chunk_gdn_{'l2' if qk_l2norm else 'raw'}_{precision}"
    op = jax_op_fn(op_name, _chunk_gdn_kernel)
    if hasattr(op, "register_fake"):

        @op.register_fake
        def _chunk_gdn_fake(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
        ) -> torch.Tensor:
            return torch.empty_like(v)

    if use_cache:
        with _LOCK:
            _GDN_OP_CACHE[cache_key] = op
    return op


def _pad_seq_dim1(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    pad_len = target_len - tensor.shape[1]
    if pad_len <= 0:
        return tensor
    if tensor.ndim == 4:
        return F.pad(tensor, (0, 0, 0, 0, 0, pad_len))
    if tensor.ndim == 3:
        return F.pad(tensor, (0, 0, 0, pad_len))
    raise ValueError(f"Expected 3D or 4D tensor for dim=1 padding, got shape {tuple(tensor.shape)}.")


def run_chunk_gdn_pallas(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    qk_l2norm: bool = True,
    buckets: tuple[int, ...] | None = DEFAULT_GDN_BUCKETS,
    precision: str = "bf16",
    op: Any = None,
) -> tuple[torch.Tensor, None]:
    """Execute causal chunked Gated DeltaNet via Pallas op with chunk-aligned static bucketing."""
    seq_len = query.shape[1]
    orig_dtype = value.dtype
    num_k_heads = query.shape[2]
    num_v_heads = value.shape[2]
    if num_k_heads != num_v_heads:
        repeat_factor = num_v_heads // num_k_heads
        query = query.repeat_interleave(repeat_factor, dim=2)
        key = key.repeat_interleave(repeat_factor, dim=2)

    compute_dtype = (
        torch.bfloat16
        if precision == "bf16" and orig_dtype in (torch.bfloat16, torch.float16)
        else orig_dtype
    )
    q_in = query.to(compute_dtype)
    k_in = key.to(compute_dtype)
    v_in = value.to(compute_dtype)
    g_in = g.to(torch.float32)
    beta_in = beta.to(torch.float32)

    bucket_len = select_gdn_bucket(seq_len, buckets=buckets)
    if bucket_len > seq_len:
        q_in = _pad_seq_dim1(q_in, bucket_len)
        k_in = _pad_seq_dim1(k_in, bucket_len)
        v_in = _pad_seq_dim1(v_in, bucket_len)
        g_in = _pad_seq_dim1(g_in, bucket_len)
        beta_in = _pad_seq_dim1(beta_in, bucket_len)

    if op is None:
        op = register_chunk_gdn_pallas_op(qk_l2norm=qk_l2norm, precision=precision)

    out = op(
        q_in.contiguous(),
        k_in.contiguous(),
        v_in.contiguous(),
        g_in.contiguous(),
        beta_in.contiguous(),
    )
    if bucket_len > seq_len:
        out = out[:, :seq_len, :, :]
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out, None


class Qwen35GdnPatchHandle:
    """Restorable handle tracking active GDN Pallas patches and call statistics."""

    def __init__(
        self,
        *,
        modeling_module: Any,
        orig_module_fn: Callable[..., Any],
        patched_fn: Callable[..., Any],
        class_patch: tuple[Any, Any, bool] | None,
        instance_patches: list[tuple[Any, bool, Any]],
        stats: dict[str, Any],
    ):
        self._modeling_module = modeling_module
        self._orig_module_fn = orig_module_fn
        self._patched_fn = patched_fn
        self._class_patch = class_patch
        self._instance_patches = instance_patches
        self.stats = stats
        self._restored = False

    def restore(self) -> None:
        """Restore the original Qwen3.5 chunk_gated_delta_rule function and module bindings."""
        if self._restored:
            return
        self._restored = True
        if getattr(self._modeling_module, "torch_chunk_gated_delta_rule", None) is self._patched_fn:
            self._modeling_module.torch_chunk_gated_delta_rule = self._orig_module_fn
        if self._class_patch is not None:
            cls, prev_attr, existed = self._class_patch
            if existed:
                cls.chunk_gated_delta_rule = prev_attr
            elif "chunk_gated_delta_rule" in cls.__dict__:
                delattr(cls, "chunk_gated_delta_rule")
        for mod, existed_in_dict, prev_val in self._instance_patches:
            if existed_in_dict:
                mod.chunk_gated_delta_rule = prev_val
            elif "chunk_gated_delta_rule" in mod.__dict__:
                delattr(mod, "chunk_gated_delta_rule")


def _fallback_reason(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    kwargs: dict[str, Any],
    require_tpu_device: bool,
) -> str | None:
    if require_tpu_device and query.device.type != "tpu":
        return "non_tpu_device"
    if initial_state is not None:
        return "initial_state"
    if output_final_state:
        return "output_final_state"
    if kwargs.get("cu_seqlens") is not None or kwargs.get("cu_seq_lens_q") is not None:
        return "cu_seqlens"
    if kwargs.get("scale") is not None:
        return "custom_scale"
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return "invalid_ndim"
    if query.shape[-1] != 128 or key.shape[-1] != 128 or value.shape[-1] != 128:
        return "head_dim_not_128"
    if query.shape[2] != key.shape[2] or query.shape[2] <= 0 or (value.shape[2] % query.shape[2]) != 0:
        return "incompatible_heads"
    return None


def install_qwen35_gdn_pallas_patch(
    model: torch.nn.Module | None = None,
    *,
    modeling_module: Any = None,
    buckets: tuple[int, ...] | None = DEFAULT_GDN_BUCKETS,
    precision: str = "bf16",
    op_resolver: Callable[..., Any] | None = None,
    validate_op: bool = True,
    require_tpu_device: bool = True,
) -> Qwen35GdnPatchHandle:
    """Patch Qwen3.5 chunked Gated DeltaNet to use deltanet-tpu Pallas kernels on TPU."""
    if modeling_module is None:
        modeling_module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")

    normalized_buckets = normalize_gdn_buckets(buckets)
    if validate_op and op_resolver is None:
        register_chunk_gdn_pallas_op(qk_l2norm=True, precision=precision)

    orig_fn = modeling_module.torch_chunk_gated_delta_rule
    orig_fn = getattr(orig_fn, "_tpu_qwen35_orig_fn", orig_fn)
    stats: dict[str, Any] = {"pallas_calls": 0, "fallback_calls": 0, "fallback_reasons": {}}

    def _patched_torch_chunk_gated_delta_rule(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs,
    ):
        reason = _fallback_reason(
            query,
            key,
            value,
            initial_state=initial_state,
            output_final_state=bool(output_final_state),
            kwargs=kwargs,
            require_tpu_device=require_tpu_device,
        )
        if reason is not None:
            stats["fallback_calls"] += 1
            reasons = stats["fallback_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
            return orig_fn(
                query,
                key,
                value,
                g,
                beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                **kwargs,
            )

        qk_l2norm = bool(use_qk_l2norm_in_kernel)
        op = (
            op_resolver(qk_l2norm=qk_l2norm, precision=precision)
            if op_resolver is not None
            else None
        )
        stats["pallas_calls"] += 1
        return run_chunk_gdn_pallas(
            query,
            key,
            value,
            g,
            beta,
            qk_l2norm=qk_l2norm,
            buckets=normalized_buckets,
            precision=precision,
            op=op,
        )

    _patched_torch_chunk_gated_delta_rule._tpu_qwen35_orig_fn = orig_fn
    modeling_module.torch_chunk_gated_delta_rule = _patched_torch_chunk_gated_delta_rule

    class_patch = None
    gdn_cls = getattr(modeling_module, "Qwen3_5GatedDeltaNet", None)
    if gdn_cls is not None:
        raw_attr = inspect.getattr_static(gdn_cls, "chunk_gated_delta_rule", None)
        existed_in_cls = "chunk_gated_delta_rule" in getattr(gdn_cls, "__dict__", {})
        if isinstance(raw_attr, staticmethod) and raw_attr.__func__ is orig_fn:
            class_patch = (gdn_cls, raw_attr, existed_in_cls)
            gdn_cls.chunk_gated_delta_rule = staticmethod(_patched_torch_chunk_gated_delta_rule)

    instance_patches: list[tuple[Any, bool, Any]] = []
    if model is not None and hasattr(model, "modules"):
        for mod in model.modules():
            if "chunk_gated_delta_rule" in getattr(mod, "__dict__", {}):
                prev_val = mod.__dict__["chunk_gated_delta_rule"]
                if prev_val is orig_fn or getattr(prev_val, "_tpu_qwen35_orig_fn", None) is orig_fn:
                    instance_patches.append((mod, True, orig_fn))
                    mod.chunk_gated_delta_rule = _patched_torch_chunk_gated_delta_rule
            else:
                bound_val = getattr(mod, "chunk_gated_delta_rule", None)
                bound_raw = inspect.getattr_static(type(mod), "chunk_gated_delta_rule", None)
                if bound_val is orig_fn or (
                    isinstance(bound_raw, staticmethod) and bound_raw.__func__ is orig_fn
                ):
                    instance_patches.append((mod, False, None))
                    mod.chunk_gated_delta_rule = _patched_torch_chunk_gated_delta_rule

    return Qwen35GdnPatchHandle(
        modeling_module=modeling_module,
        orig_module_fn=orig_fn,
        patched_fn=_patched_torch_chunk_gated_delta_rule,
        class_patch=class_patch,
        instance_patches=instance_patches,
        stats=stats,
    )


@contextmanager
def qwen35_gdn_pallas_context(enabled: bool = True, **kwargs):
    """Context manager enabling the Qwen3.5 GDN Pallas patch when enabled=True."""
    if not enabled:
        yield None
        return
    handle = install_qwen35_gdn_pallas_patch(**kwargs)
    try:
        yield handle
    finally:
        handle.restore()


def prompt_cache_path(
    cache_dir: str | Path,
    model_id: str,
    prompt: str,
    max_sequence_length: int,
) -> Path:
    """Return the deterministic .pt file path for a cached text embedding entry."""
    payload = json.dumps(
        {
            "v": PROMPT_CACHE_VERSION,
            "model": str(model_id),
            "max_sequence_length": int(max_sequence_length),
            "prompt": prompt,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return Path(cache_dir) / f"{digest}.pt"


def load_cached_prompt_embed(
    cache_dir: str | Path,
    model_id: str,
    prompt: str,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load and validate a persisted prompt embedding [L, D] and mask [L], or return None."""
    path = prompt_cache_path(cache_dir, model_id, prompt, max_sequence_length)
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != PROMPT_CACHE_VERSION:
        return None
    if payload.get("model") != str(model_id):
        return None
    if payload.get("prompt") != prompt:
        return None
    if payload.get("max_sequence_length") != int(max_sequence_length):
        return None
    embeds = payload.get("embeds")
    mask = payload.get("mask")
    if not isinstance(embeds, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return None
    if embeds.ndim != 2 or mask.ndim != 1 or embeds.shape[0] != mask.shape[0] or embeds.shape[0] == 0:
        return None
    return embeds.detach().cpu(), mask.detach().cpu().to(torch.bool)


def save_cached_prompt_embed(
    cache_dir: str | Path,
    model_id: str,
    prompt: str,
    max_sequence_length: int,
    embeds: torch.Tensor,
    mask: torch.Tensor,
) -> Path:
    """Atomically persist a prompt embedding [L, D] (or [1, L, D]) and mask [L] to disk."""
    embeds_cpu = embeds[0] if embeds.ndim == 3 and embeds.shape[0] == 1 else embeds
    mask_cpu = mask[0] if mask.ndim == 2 and mask.shape[0] == 1 else mask
    if embeds_cpu.ndim != 2 or mask_cpu.ndim != 1 or embeds_cpu.shape[0] != mask_cpu.shape[0]:
        raise ValueError(
            "Expected embeds shape [L, D] (or [1, L, D]) and mask shape [L] (or [1, L]), "
            f"got {tuple(embeds.shape)} and {tuple(mask.shape)}."
        )
    target = prompt_cache_path(cache_dir, model_id, prompt, max_sequence_length)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd_num, tmp_path = tempfile.mkstemp(prefix=target.stem + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd_num, "wb") as handle:
            torch.save(
                {
                    "version": PROMPT_CACHE_VERSION,
                    "model": str(model_id),
                    "prompt": prompt,
                    "max_sequence_length": int(max_sequence_length),
                    "embeds": embeds_cpu.detach().cpu().contiguous(),
                    "mask": mask_cpu.detach().cpu().to(torch.bool).contiguous(),
                },
                handle,
            )
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return target
