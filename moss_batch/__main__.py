"""Run encoding and generation as separate processes: python -m moss_batch."""
from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

from moss_batch.jobs import load_jobs


@contextmanager
def directory_lock(directory):
    """An interrupted process leaves a visible lock; never guess another writer is dead."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".moss-batch.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"{lock} exists. Check no writer is running before removing a stale lock.") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid()}, handle)
        yield
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("encode", "generate", "validate"))
    parser.add_argument("--jobs", type=Path, required=True, help="UTF-8 JSONL task manifest")
    parser.add_argument("--model", help="Official Hub model ID or local model bundle")
    parser.add_argument("--revision", help="Pin a Hub model revision (commit recommended)")
    parser.add_argument("--local-model-version", help="Portable local bundle version; change it when files change")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("encodings"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--offload", choices=("none", "model", "sequential"), default="model")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    try:
        jobs = load_jobs(args.jobs)
        if args.stage == "validate":
            print(f"Valid manifest: {len(jobs)} job(s)")
            return 0
        if not args.model:
            raise ValueError("--model is required for encoding and generation.")
        from moss_batch.runtime import encode_jobs, generate_jobs, resolve_model

        source, identity = resolve_model(
            args.model, args.stage, args.revision, args.local_model_version, args.local_files_only,
        )
        print("Model: " + json.dumps(identity, ensure_ascii=False))
        directory = args.cache_dir if args.stage == "encode" else args.output_dir
        with directory_lock(directory):
            if args.stage == "encode":
                encode_jobs(jobs, source, identity, args.cache_dir, args.offload, args.device)
            else:
                generate_jobs(jobs, source, identity, args.cache_dir, args.output_dir, args.offload, args.device)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
