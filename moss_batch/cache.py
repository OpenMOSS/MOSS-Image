"""Portable, validated prompt encodings stored without pickle or model weights."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

SCHEMA_VERSION = 1
TENSOR_FILENAME = "encoding.safetensors"
METADATA_FILENAME = "metadata.json"
_REQUIRED = {"prompt_embeds", "prompt_attention_mask"}
_NEGATIVE = {"negative_prompt_embeds", "negative_prompt_attention_mask"}


class EncodingCacheError(ValueError):
    """An encoding is malformed or incompatible with the requested provenance."""


def _validate_tensors(tensors: dict[str, torch.Tensor]) -> None:
    if not isinstance(tensors, dict):
        raise EncodingCacheError("Encoding tensors must be a dictionary.")
    keys = set(tensors)
    if not _REQUIRED <= keys or keys - (_REQUIRED | _NEGATIVE):
        raise EncodingCacheError(
            "Encoding requires prompt_embeds and prompt_attention_mask; "
            "only the optional negative_prompt_embeds/negative_prompt_attention_mask pair is allowed."
        )
    if keys & _NEGATIVE and not _NEGATIVE <= keys:
        raise EncodingCacheError("Negative prompt embeddings and attention mask must be supplied together.")
    for prefix in ("prompt", "negative_prompt"):
        if f"{prefix}_embeds" not in tensors:
            continue
        embeds, mask = tensors[f"{prefix}_embeds"], tensors[f"{prefix}_attention_mask"]
        if not isinstance(embeds, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise EncodingCacheError(f"{prefix} embeddings and attention mask must be torch tensors.")
        if embeds.layout != torch.strided or mask.layout != torch.strided:
            raise EncodingCacheError(f"{prefix} tensors must be dense strided tensors.")
        if embeds.ndim != 3 or embeds.shape[0] != 1 or min(embeds.shape[1:]) < 1:
            raise EncodingCacheError(f"{prefix}_embeds must have shape [1, sequence_length, hidden_size].")
        if embeds.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise EncodingCacheError(f"{prefix}_embeds must use float16, bfloat16, or float32.")
        if not bool(torch.isfinite(embeds).all()):
            raise EncodingCacheError(f"{prefix}_embeds contains NaN or infinity; re-encode this prompt.")
        if mask.ndim != 2 or tuple(mask.shape) != tuple(embeds.shape[:2]):
            raise EncodingCacheError(f"{prefix}_attention_mask must match embeddings shape [1, sequence_length].")
        if mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise EncodingCacheError(f"{prefix}_attention_mask must use a boolean or integer dtype.")
        if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.any()):
            raise EncodingCacheError(f"{prefix}_attention_mask must contain only 0/1 and attend to at least one token.")
    if _NEGATIVE <= keys:
        positive, negative = tensors["prompt_embeds"], tensors["negative_prompt_embeds"]
        if positive.shape[-1] != negative.shape[-1] or positive.dtype != negative.dtype:
            raise EncodingCacheError("Positive and negative embeddings must have the same hidden size and dtype.")


def _metadata_json(metadata: dict[str, Any]) -> str:
    if not isinstance(metadata, dict):
        raise EncodingCacheError("Encoding metadata must be a JSON object.")
    try:
        encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        # Reject tuples, non-string keys, and other values that silently change on export.
        if json.loads(encoded) != metadata:
            raise ValueError("metadata changes when serialized as JSON")
        return json.dumps(
            {"schema_version": SCHEMA_VERSION, "metadata": metadata},
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n"
    except (TypeError, ValueError, OverflowError) as exc:
        raise EncodingCacheError(f"Encoding metadata must contain JSON values only: {exc}") from exc


def _same_json_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_json_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json_value(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _check_expected(actual: Any, expected: Any, location: str = "metadata") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise EncodingCacheError(f"Incompatible encoding: {location} is not an object.")
        for key, value in expected.items():
            if key not in actual:
                raise EncodingCacheError(f"Incompatible encoding: missing {location}.{key}; re-encode this prompt.")
            _check_expected(actual[key], value, f"{location}.{key}")
    elif not _same_json_value(actual, expected):
        raise EncodingCacheError(
            f"Incompatible encoding: {location} differs from the requested value; re-encode this prompt."
        )


def save_encoding(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    """Publish a new encoding directory atomically; existing paths are never reused.

    Copies are saved on CPU in safetensors format. Metadata is a JSON object chosen
    by the caller, which should include model, tokenizer, template and prompt
    provenance. An existing destination raises FileExistsError, even if corrupt.
    """
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f"Encoding destination already exists: {path}")
    metadata_text = _metadata_json(metadata)
    _validate_tensors(tensors)
    # Clone to break shared storage (safetensors rejects aliases) and detach autograd.
    cpu_tensors = {name: tensor.detach().to("cpu").contiguous().clone() for name, tensor in tensors.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))
    try:
        save_file(cpu_tensors, str(temporary / TENSOR_FILENAME))
        (temporary / METADATA_FILENAME).write_text(metadata_text, encoding="utf-8")
        if os.path.lexists(path):
            raise FileExistsError(f"Encoding destination already exists: {path}")
        # Other writers publish nonempty directories, so a concurrent publication
        # cannot be overwritten by rename on either Windows or POSIX.
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def load_encoding(path: Path, expected: dict[str, Any] | None = None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load and validate CPU tensors, optionally checking a provenance subset.

    ``expected`` recursively compares selected metadata fields. Lists and scalar
    values must match exactly, including their JSON type. No prompt is rewritten.
    Corruption or a mismatch raises EncodingCacheError with the failing field.
    """
    path = Path(path)
    if expected is not None and not isinstance(expected, dict):
        raise EncodingCacheError("Expected provenance must be a dictionary.")
    try:
        envelope = json.loads(
            (path / METADATA_FILENAME).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise EncodingCacheError(f"Cannot read encoding metadata at {path}: {exc}") from exc
    if not isinstance(envelope, dict) or type(envelope.get("schema_version")) is not int:
        raise EncodingCacheError(f"Invalid encoding metadata envelope at {path}.")
    if envelope["schema_version"] != SCHEMA_VERSION:
        raise EncodingCacheError(
            f"Unsupported encoding schema {envelope['schema_version']} at {path}; expected {SCHEMA_VERSION}."
        )
    metadata = envelope.get("metadata")
    if not isinstance(metadata, dict):
        raise EncodingCacheError(f"Encoding metadata at {path} must be a JSON object.")
    if expected is not None:
        _check_expected(metadata, expected)
    try:
        tensors = load_file(str(path / TENSOR_FILENAME), device="cpu")
    except (OSError, SafetensorError, RuntimeError, ValueError) as exc:
        raise EncodingCacheError(f"Cannot read safetensors encoding at {path}: {exc}") from exc
    _validate_tensors(tensors)
    return tensors, metadata
