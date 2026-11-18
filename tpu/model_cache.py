"""RAM-backed checkpoint downloads for the TPU runtime."""

from __future__ import annotations

import errno
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from filelock import FileLock
from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_OFFLINE
from huggingface_hub.utils import LocalEntryNotFoundError, filter_repo_objects

_LOG = logging.getLogger(__name__)
_COLAB_RAM = Path("/content/.moss-image-tpu-ram")
_SHARED_RAM = Path("/dev/shm")
_GIB = 1024**3
_HOST_HEADROOM = 8 * _GIB
_COMPONENT_MARKERS = ".moss-image-components"


def _is_tmpfs(path: Path) -> bool:
    for entry in Path("/proc/self/mounts").read_text().splitlines():
        _, mountpoint, filesystem, *_ = entry.split()
        if mountpoint == str(path) and filesystem == "tmpfs":
            return True
    return False


def _ram_cache(create: bool) -> Path | None:
    try:
        if create and os.getenv("COLAB_RELEASE_TAG") and os.geteuid() == 0 and _COLAB_RAM.parent.is_dir():
            with FileLock(str(_COLAB_RAM) + ".lock"):
                if not (_COLAB_RAM.is_mount() or _COLAB_RAM.is_symlink()):
                    _COLAB_RAM.mkdir(mode=0o700, exist_ok=True)
                    # Colab's default /dev/shm can be smaller than the full checkpoint.
                    if not any(_COLAB_RAM.iterdir()):
                        subprocess.run(
                            ["mount", "-t", "tmpfs", "-o", "size=30G,nosuid,nodev,noexec,mode=0700",
                             "tmpfs", str(_COLAB_RAM)],
                            check=False, capture_output=True,
                        )
        for mount in (_COLAB_RAM, _SHARED_RAM):
            if not _is_tmpfs(mount):
                continue
            cache = mount / f"moss-image-hub-{os.geteuid()}"
            if create:
                cache.mkdir(mode=0o700, exist_ok=True)
            if cache.is_dir() and not cache.is_symlink() and cache.stat().st_uid == os.geteuid():
                return cache
    except OSError as exc:
        _LOG.debug("TPU RAM cache unavailable: %s", exc)
    return None


def download_to_ram(
    source: str, patterns: list[str], components: tuple[str, ...], available_gib: float | None,
    *, local_files_only: bool,
) -> Path | None:
    if sys.platform != "linux" or os.getenv("MOSS_IMAGE_TPU_DISK_CACHE") == "1":
        return None
    offline = local_files_only or HF_HUB_OFFLINE
    cache = _ram_cache(create=not offline)
    if cache is None:
        return None
    options = {"cache_dir": cache, "allow_patterns": patterns}
    if offline:
        try:
            path = Path(snapshot_download(source, local_files_only=True, **options))
        except LocalEntryNotFoundError:
            return None
        if (path / "pipeline_moss_image.py").is_file() and all(
            (path / _COMPONENT_MARKERS / name).is_file() for name in components
        ):
            return path
        return None

    with FileLock(str(cache / ".moss-image-download.lock")):
        try:
            info = HfApi().model_info(source, files_metadata=True, timeout=10)
        except Exception as exc:
            # Size lookup is optional; the normal Hub loader handles the request and its errors.
            _LOG.debug("Could not size the TPU RAM download: %s", exc)
            return None
        files = {
            file.rfilename: file.size
            for file in filter_repo_objects(info.siblings or (), allow_patterns=patterns,
                                            key=lambda file: file.rfilename)
        }
        if not files or any(size is None for size in files.values()):
            return None
        missing_bytes = sum(
            size for name, size in files.items()
            if not isinstance(try_to_load_from_cache(source, name, cache_dir=cache, revision=info.sha), str)
        )
        free_ram = max(0, int((available_gib or 0) * _GIB) - _HOST_HEADROOM)
        if missing_bytes > min(free_ram, shutil.disk_usage(cache).free):
            _LOG.warning("Insufficient space for the TPU model in RAM; using the disk cache.")
            return None
        try:
            # Download to tmpfs so the first TPU upload never waits for disk-backed weight pages.
            path = Path(snapshot_download(source, **options))
        except (OSError, RuntimeError) as exc:
            if getattr(exc, "errno", None) not in (errno.ENOSPC, errno.EDQUOT) and "No space left on device" not in str(exc):
                raise
            _LOG.warning("TPU RAM cache filled during download; using the disk cache.")
            return None
        if not all((path / name).is_file() for name in files):
            return None
        markers = path / _COMPONENT_MARKERS
        markers.mkdir(exist_ok=True)
        for name in components:
            (markers / name).touch()
        return path
