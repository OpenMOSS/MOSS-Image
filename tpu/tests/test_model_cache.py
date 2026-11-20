import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tpu import model_cache, tpu_runtime


class ModelCacheTests(unittest.TestCase):
    model = "owner/model"
    patterns = ["pipeline_moss_image.py", "transformer/*"]
    components = ("transformer",)

    def test_normal_tpu_resolution_prefers_ram(self):
        with (
            patch.object(tpu_runtime, "is_available", return_value=True),
            patch.object(model_cache, "download_to_ram", return_value=Path("ram-snapshot")) as ram,
            patch.object(tpu_runtime, "snapshot_download") as disk,
        ):
            self.assertEqual(tpu_runtime.resolve_model_bundle(self.model), Path("ram-snapshot"))
            disk.assert_not_called()
            self.assertIn("text_encoder/*", ram.call_args.args[1])

    def test_offline_prefers_only_complete_ram_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pipeline_moss_image.py").touch()
            with (
                patch.object(model_cache.sys, "platform", "linux"),
                patch.dict(model_cache.os.environ, {"MOSS_IMAGE_TPU_DISK_CACHE": "0"}),
                patch.object(model_cache, "_ram_cache", return_value=root) as cache,
                patch.object(model_cache, "snapshot_download", return_value=root) as download,
                patch.object(model_cache, "HfApi") as api,
            ):
                args = (self.model, self.patterns, self.components, 1)
                self.assertIsNone(model_cache.download_to_ram(*args, local_files_only=True))
                cache.assert_called_once_with(create=False)
                marker = root / model_cache._COMPONENT_MARKERS
                marker.mkdir()
                (marker / "transformer").touch()
                self.assertEqual(model_cache.download_to_ram(*args, local_files_only=True), root)
                self.assertTrue(download.call_args.kwargs["local_files_only"])
                api.assert_not_called()

    def test_ram_download_reserves_only_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pipeline_moss_image.py").touch()
            (root / "transformer").mkdir()
            (root / "transformer/weights.safetensors").touch()
            files = [
                SimpleNamespace(rfilename="pipeline_moss_image.py", size=1024),
                SimpleNamespace(rfilename="transformer/weights.safetensors", size=5 * 1024**3),
                SimpleNamespace(rfilename="unused", size=None),
            ]
            with (
                patch.object(model_cache.sys, "platform", "linux"),
                patch.dict(model_cache.os.environ, {"MOSS_IMAGE_TPU_DISK_CACHE": "0"}),
                patch.object(model_cache, "HF_HUB_OFFLINE", False),
                patch.object(model_cache, "_ram_cache", return_value=root),
                patch.object(model_cache, "HfApi") as api,
                patch.object(model_cache, "try_to_load_from_cache", return_value=None) as cached,
                patch.object(model_cache, "snapshot_download", return_value=root) as download,
            ):
                api.return_value.model_info.return_value = SimpleNamespace(siblings=files, sha="sha")
                args = (self.model, self.patterns, self.components, 9)
                self.assertIsNone(model_cache.download_to_ram(*args, local_files_only=False))
                download.assert_not_called()
                cached.return_value = str(root)
                self.assertEqual(model_cache.download_to_ram(*args, local_files_only=False), root)
                self.assertEqual(download.call_args.kwargs["cache_dir"], root)
                self.assertTrue((root / model_cache._COMPONENT_MARKERS / "transformer").is_file())
