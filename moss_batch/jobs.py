"""Validated JSONL jobs and resumable, single-writer generation outputs.

The CLI owns the output-root lock. Call ``prepare`` before inference: random
seeds are persisted before any work, so an interrupted job can be replayed.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PARAMETERS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 3.0,
    "shift": 5.0,
    "max_sequence_length": 4096,
    "uncond_skip_layers": [22],
    "slg_layers": [],
    "slg_scale": 0.0,
    "slg_start": 0.0,
    "slg_stop": 1.0,
}
_RESERVED_IDS = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10)
}


def _integer(value: Any, name: str, minimum: int, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer >= {minimum}" +
                         (f" and <= {maximum}" if maximum is not None else ""))
    return value


def _parameters(values: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError("Job parameters must be an object")
    unknown = set(values) - DEFAULT_PARAMETERS.keys()
    if unknown:
        raise ValueError(f"Unknown job parameters: {', '.join(sorted(unknown))}")
    result = {**DEFAULT_PARAMETERS, **values}
    for name in ("height", "width"):
        if _integer(result[name], name, 16) % 16:
            raise ValueError(f"{name} must be a multiple of 16")
    _integer(result["num_inference_steps"], "num_inference_steps", 1)
    _integer(result["max_sequence_length"], "max_sequence_length", 1, 4096)
    for name in ("guidance_scale", "shift", "slg_scale", "slg_start", "slg_stop"):
        value = result[name]
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        result[name] = float(value)
    if result["shift"] <= 0:
        raise ValueError("shift must be positive")
    if not 0 <= result["slg_start"] <= result["slg_stop"] <= 1:
        raise ValueError("Require 0 <= slg_start <= slg_stop <= 1")
    for name in ("uncond_skip_layers", "slg_layers"):
        values = result[name]
        if not isinstance(values, list):
            raise ValueError(f"{name} must be an array of unique nonnegative integers")
        for value in values:
            _integer(value, name, 0)
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must contain unique indices")
        result[name] = list(values)
    return result


@dataclass(frozen=True)
class Job:
    id: str
    prompt: str
    negative_prompt: str = ""
    seed: int = -1
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (not isinstance(self.id, str) or
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", self.id) is None or
                self.id.upper() in _RESERVED_IDS):
            raise ValueError("id must be a portable filename: 1-128 ASCII letters, digits, '_' or '-', starting with a letter or digit; reserved device names are forbidden")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if not isinstance(self.negative_prompt, str):
            raise ValueError("negative_prompt must be a string")
        _integer(self.seed, "seed", -1, 2**53 - 1)
        object.__setattr__(self, "parameters", _parameters(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "prompt": self.prompt,
                "negative_prompt": self.negative_prompt, "seed": self.seed,
                **self.parameters}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def load_jobs(path: str | Path) -> list[Job]:
    """Read flat JSONL records; blank lines are ignored, duplicate IDs rejected."""
    jobs = []
    ids = set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_unique_object)
                if not isinstance(row, dict):
                    raise ValueError("Each line must be an object")
                missing = {"id", "prompt"} - row.keys()
                if missing:
                    raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")
                arguments = {name: row.pop(name) for name in
                             ("id", "prompt", "negative_prompt", "seed") if name in row}
                job = Job(**arguments, parameters=row)
                if job.id.casefold() in ids:
                    raise ValueError(f"Duplicate job id (case insensitive): {job.id}")
                ids.add(job.id.casefold())
                jobs.append(job)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not jobs:
        raise ValueError(f"{path}: no jobs found")
    return jobs


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".parameters-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _valid_image(path: Path, size: tuple[int, int]) -> bool:
    from PIL import Image

    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != size:
                return False
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


class JobOutput:
    """Per-job record; callers must serialize writers with an output-root lock."""

    def __init__(self, output_root: str | Path, job: Job, provenance: dict[str, Any]):
        self.job = job
        self.directory = Path(output_root) / job.id
        self.record_path = self.directory / "parameters.json"
        self.image_path = self.directory / "01.png"
        # Snapshot the exact JSON representation so later caller mutations cannot
        # silently change the resume identity.
        self.identity = json.loads(json.dumps(
            {"job": job.to_dict(), "provenance": provenance}, allow_nan=False))
        self.image_size = (job.parameters["width"], job.parameters["height"])

    def _read(self) -> dict | None:
        if not self.record_path.exists():
            return None
        try:
            with self.record_path.open(encoding="utf-8") as stream:
                record = json.load(stream, object_pairs_hook=_unique_object)
            if (not isinstance(record, dict) or
                    type(record.get("schema_version")) is not int or record["schema_version"] != 1):
                raise ValueError("Unsupported output record")
            # Python considers True == 1 and 1 == 1.0; resume identities must
            # retain JSON types as well as values, independent of key ordering.
            if any(json.dumps(record.get(key), sort_keys=True, allow_nan=False) !=
                   json.dumps(value, sort_keys=True, allow_nan=False)
                   for key, value in self.identity.items()):
                raise ValueError("Job settings or model/cache provenance differ from the existing output")
            _integer(record.get("actual_seed"), "actual_seed", 0, 2**53 - 1)
            if self.job.seed != -1 and record["actual_seed"] != self.job.seed:
                raise ValueError("Stored actual_seed does not match requested seed")
            if record.get("status") not in ("pending", "complete"):
                raise ValueError("Invalid job status")
            if record["status"] == "complete" and record.get("image") != "01.png":
                raise ValueError("Invalid recorded image filename")
            return record
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Cannot resume {self.record_path}: {exc}") from exc

    def prepare(self) -> int:
        """Persist a seed before work; an existing matching job retains its seed."""
        record = self._read()
        if record is not None:
            return record["actual_seed"]
        if self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError(f"Refusing untracked nonempty output directory: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        seed = secrets.randbelow(2**53) if self.job.seed == -1 else self.job.seed
        _atomic_json(self.record_path, {"schema_version": 1, **self.identity,
                                      "actual_seed": seed, "status": "pending"})
        return seed

    def is_complete(self) -> bool:
        record = self._read()
        return bool(record is not None and record["status"] == "complete"
                    and _valid_image(self.image_path, self.image_size))

    def complete(self, image) -> Path:
        """Publish a valid PNG, then mark complete; failed writes remain resumable."""
        record = self._read()
        if record is None:
            raise ValueError("Call prepare() before complete()")
        if self.is_complete():
            return self.image_path
        descriptor, temporary = tempfile.mkstemp(prefix=".image-", suffix=".png", dir=self.directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                image.save(stream, format="PNG")
                stream.flush()
                os.fsync(stream.fileno())
            if not _valid_image(Path(temporary), self.image_size):
                raise ValueError("Generated image must be a readable PNG with the requested dimensions")
            os.replace(temporary, self.image_path)
            _atomic_json(self.record_path, {**record, "status": "complete", "image": "01.png"})
        finally:
            Path(temporary).unlink(missing_ok=True)
        return self.image_path
