"""Checkpoint key normalization for the native ComfyUI loaders."""

import re

_FUSED_BLOCK_QKV = re.compile(r"^layers\.(\d+)\.proj\.qkv\.(.+)$")

_FUSED_PARTS = ("q", "k", "v")


def unfuse_block_qkv(state):
    """Split fused ``layers.N.proj.qkv`` tensors into ``proj.{q,k,v}``.

    The reference MOSS-Image transformer keeps self-attention projections
    separate, so the diffusers checkpoints and ``export.py`` ship
    ``proj.q``/``proj.k``/``proj.v``. Comfy-Org single-file checkpoints instead
    pack them into one ``proj.qkv`` row block, which ``_Block`` has no attribute
    for. Splitting here keeps the model definition training-compatible while
    accepting both layouts.

    Only the DiT blocks are rewritten: ``text_embedder.blocks.N.qkv`` is fused
    in every checkpoint and ``_TextBlock`` already expects it that way.

    Returns the same mapping, mutated in place.
    """
    fused = {}
    for key in list(state):
        match = _FUSED_BLOCK_QKV.match(key)
        if match is not None:
            fused.setdefault(match.group(1), {})[match.group(2)] = key
    for index, tensors in fused.items():
        if "weight" not in tensors:
            raise ValueError(f"layers.{index}.proj.qkv has no weight tensor")
        weight = state[tensors["weight"]]
        if weight.ndim != 2 or weight.shape[0] % len(_FUSED_PARTS):
            raise ValueError(f"layers.{index}.proj.qkv.weight is not a fused 2-D projection")
        if all(f"layers.{index}.proj.{name}.weight" in state for name in _FUSED_PARTS):
            continue
        for suffix in tensors:
            value = state[tensors[suffix]]
            if suffix == "comfy_quant":
                # Per-layer quantization metadata is identical for each slice.
                for name in _FUSED_PARTS:
                    state[f"layers.{index}.proj.{name}.comfy_quant"] = value
                continue
            if value.shape[0] != weight.shape[0]:
                raise ValueError(f"layers.{index}.proj.qkv.{suffix} does not match the fused weight")
            chunks = value.chunk(len(_FUSED_PARTS), 0)
            for name, chunk in zip(_FUSED_PARTS, chunks):
                state[f"layers.{index}.proj.{name}.{suffix}"] = chunk.contiguous()
        for key in tensors.values():
            del state[key]
    return state
