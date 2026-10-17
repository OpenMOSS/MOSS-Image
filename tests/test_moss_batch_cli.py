import builtins
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from moss_batch.__main__ import directory_lock, main


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "jobs.jsonl"

    @staticmethod
    def forbid_model_imports(name, *args, **kwargs):
        if (name == "moss_batch.runtime" or name.split(".")[0] in
                {"torch", "diffusers", "transformers", "huggingface_hub", "safetensors", "PIL"}):
            raise AssertionError(f"Manifest validation imported inference dependency: {name}")
        return _real_import(name, *args, **kwargs)

    def test_validate_needs_no_model_or_inference_dependencies(self):
        self.manifest.write_text(json.dumps({"id": "one", "prompt": "city"}), encoding="utf-8")
        stdout = io.StringIO()
        with patch("builtins.__import__", side_effect=self.forbid_model_imports), redirect_stdout(stdout):
            result = main(["validate", "--jobs", str(self.manifest)])
        self.assertEqual(result, 0)
        self.assertIn("1 job(s)", stdout.getvalue())
        self.assertEqual([p.name for p in self.root.iterdir()], ["jobs.jsonl"])

    def test_invalid_manifest_fails_before_model_resolution(self):
        self.manifest.write_text('{"id":"one","prompt":"city","width":17}', encoding="utf-8")
        for stage in ("validate", "encode", "generate"):
            with self.subTest(stage=stage):
                stderr = io.StringIO()
                with patch("builtins.__import__", side_effect=self.forbid_model_imports), redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as caught:
                        main([stage, "--jobs", str(self.manifest), "--model", "unavailable/model"])
                self.assertEqual(caught.exception.code, 1)
                self.assertIn("width must be a multiple of 16", stderr.getvalue())

    def test_lock_refuses_second_writer_without_removing_first_lock(self):
        directory = self.root / "outputs"
        lock = directory / ".moss-batch.lock"
        with directory_lock(directory):
            self.assertTrue(lock.is_file())
            self.assertIsInstance(json.loads(lock.read_text(encoding="utf-8"))["pid"], int)
            with self.assertRaisesRegex(ValueError, "exists"):
                with directory_lock(directory):
                    self.fail("Second writer acquired the same output directory")
            self.assertTrue(lock.exists())
        self.assertFalse(lock.exists())

    def test_lock_cleaned_up_after_exception(self):
        directory = self.root / "outputs"
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with directory_lock(directory):
                raise RuntimeError("generation failed")
        self.assertFalse((directory / ".moss-batch.lock").exists())
        with directory_lock(directory):
            pass


_real_import = builtins.__import__


if __name__ == "__main__":
    unittest.main()
