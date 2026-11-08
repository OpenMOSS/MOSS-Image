import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from moss_batch.jobs import Job, JobOutput, load_jobs


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def load(self, rows):
        path = self.root / "jobs.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        return load_jobs(path)

    def test_defaults_and_unicode(self):
        job, = self.load([{"id": "poster-1", "prompt": "城市 🌆"}])
        self.assertEqual(job.prompt, "城市 🌆")
        self.assertEqual(job.parameters["num_inference_steps"], 50)
        self.assertEqual(job.parameters["uncond_skip_layers"], [22])
        self.assertEqual(job.to_dict()["shift"], 5)
        self.assertEqual(job.seed, -1)

    def test_custom_parameters(self):
        job, = self.load([{"id": "1", "prompt": "p", "seed": 0,
                           "height": 512, "guidance_scale": 0,
                           "uncond_skip_layers": [], "slg_layers": [1, 2]}])
        self.assertEqual(job.parameters["height"], 512)
        self.assertEqual(job.parameters["slg_layers"], [1, 2])
        self.assertEqual(job.seed, 0)

    def test_ids_safe_on_windows_and_posix(self):
        for value in ("../x", "x/y", "x\\y", "/x", "C:x", ".", "..", "x.",
                      "x ", "NUL", "con", "LPT1", "COM9", "", "a" * 129):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Job(value, "prompt")

    def test_invalid_settings(self):
        for key, value in (("height", 0), ("width", 1000), ("height", True),
                           ("num_inference_steps", 0), ("num_inference_steps", 3.5),
                           ("guidance_scale", float("nan")), ("shift", float("inf")),
                           ("guidance_scale", 10**1000),
                           ("shift", 0), ("slg_scale", -1), ("slg_start", 2),
                           ("max_sequence_length", 4097), ("uncond_skip_layers", "22"),
                           ("slg_layers", [1, 1]), ("slg_layers", [True]),
                           ("slg_layers", [-1]), ("guidance_scale", True),
                           ("num_images_per_prompt", 2), ("unknown", 1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.load([{"id": "x", "prompt": "p", key: value}])
        for seed in (-2, 2**53, True, "5"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                Job("x", "p", seed=seed)
        with self.assertRaises(ValueError):
            Job("x", "p", parameters={"slg_start": 0.7, "slg_stop": 0.2})

    def test_prompt_and_schema_errors_report_line(self):
        for row in ({"id": "x"}, {"prompt": "p"}, {"id": "x", "prompt": " "},
                    {"id": "x", "prompt": 42}, {"id": "x", "prompt": "p", "negative_prompt": None}, []):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, r"jobs.jsonl:1:"):
                self.load([row])

    def test_duplicate_ids_including_case(self):
        with self.assertRaisesRegex(ValueError, "Duplicate job id"):
            self.load([{"id": "A", "prompt": "one"}, {"id": "a", "prompt": "two"}])

    def test_duplicate_fields_rejected(self):
        path = self.root / "jobs.jsonl"
        path.write_text('{"id":"x","prompt":"one","prompt":"two"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON field"):
            load_jobs(path)

    def test_bom_blank_lines_and_empty(self):
        path = self.root / "jobs.jsonl"
        path.write_text('\n{"id":"x","prompt":"p"}\n\n', encoding="utf-8-sig")
        self.assertEqual(len(load_jobs(path)), 1)
        path.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no jobs"):
            load_jobs(path)


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job = Job("city", "城市 🌆", parameters={"width": 16, "height": 16})
        self.provenance = {"model": "OpenMOSS/model", "revision": "abc", "cache": "encoding-1"}
        self.output = JobOutput(self.root, self.job, self.provenance)

    def test_random_seed_persisted_before_work_and_reused(self):
        self.assertFalse(self.output.is_complete())
        with patch("moss_batch.jobs.secrets.randbelow", return_value=123):
            self.assertEqual(self.output.prepare(), 123)
        resumed = JobOutput(self.root, self.job, self.provenance)
        self.assertEqual(resumed.prepare(), 123)
        record = json.loads(self.output.record_path.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "pending")
        self.assertFalse(self.output.is_complete())

    def test_successful_publication_and_idempotence(self):
        self.output.prepare()
        expected = Image.new("RGB", (16, 16), "red")
        self.assertEqual(self.output.complete(expected), self.output.image_path)
        self.assertTrue(self.output.is_complete())
        self.output.complete(Image.new("RGB", (16, 16), "blue"))
        with Image.open(self.output.image_path) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(JobOutput(self.root, self.job, self.provenance).prepare(), self.output.prepare())

    def test_failure_during_save_remains_pending(self):
        class BrokenImage:
            def save(self, stream, format):
                stream.write(b"partial")
                raise OSError("interrupted")
        seed = self.output.prepare()
        with self.assertRaisesRegex(OSError, "interrupted"):
            self.output.complete(BrokenImage())
        self.assertFalse(self.output.is_complete())
        self.assertFalse(self.output.image_path.exists())
        self.assertEqual(self.output.prepare(), seed)
        self.assertEqual([p.name for p in self.output.directory.iterdir()], ["parameters.json"])

    def test_image_published_but_record_write_interrupted_retries(self):
        seed = self.output.prepare()
        with patch("moss_batch.jobs._atomic_json", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.output.complete(Image.new("RGB", (16, 16)))
        self.assertTrue(self.output.image_path.exists())
        self.assertFalse(self.output.is_complete())
        self.assertEqual(self.output.prepare(), seed)
        self.output.complete(Image.new("RGB", (16, 16)))
        self.assertTrue(self.output.is_complete())

    def test_deleted_or_corrupt_completed_image_retries_same_seed(self):
        seed = self.output.prepare()
        self.output.complete(Image.new("RGB", (16, 16)))
        self.output.image_path.write_bytes(b"not a PNG")
        self.assertFalse(self.output.is_complete())
        self.assertEqual(self.output.prepare(), seed)
        self.output.complete(Image.new("RGB", (16, 16)))
        self.output.image_path.unlink()
        self.assertFalse(self.output.is_complete())
        self.assertEqual(self.output.prepare(), seed)

    def test_settings_and_provenance_cannot_be_overwritten(self):
        self.output.prepare()
        for job, provenance in ((Job("city", "different"), self.provenance),
                                (self.job, {**self.provenance, "revision": "new"}),
                                (Job("city", self.job.prompt, parameters={"width": 512}), self.provenance)):
            with self.subTest(job=job, provenance=provenance):
                output = JobOutput(self.root, job, provenance)
                with self.assertRaisesRegex(ValueError, "differ"):
                    output.prepare()
                with self.assertRaisesRegex(ValueError, "differ"):
                    output.is_complete()

    def test_untracked_output_not_overwritten(self):
        self.output.directory.mkdir()
        self.output.image_path.write_bytes(b"existing")
        with self.assertRaisesRegex(ValueError, "untracked"):
            self.output.prepare()
        self.assertEqual(self.output.image_path.read_bytes(), b"existing")

    def test_explicit_seed_and_prepare_required(self):
        with self.assertRaisesRegex(ValueError, "prepare"):
            self.output.complete(Image.new("RGB", (16, 16)))
        output = JobOutput(self.root, Job("zero", "p", seed=0), self.provenance)
        self.assertEqual(output.prepare(), 0)

    def test_invalid_record_is_not_overwritten(self):
        self.output.prepare()
        self.output.record_path.write_text("not json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.output.prepare()
        self.assertEqual(self.output.record_path.read_text(encoding="utf-8"), "not json")

    def test_boolean_schema_and_integer_fields_rejected(self):
        self.output.prepare()
        original = json.loads(self.output.record_path.read_text(encoding="utf-8"))
        for key, value in (("schema_version", True), ("schema_version", 1.0),
                           ("actual_seed", True), ("actual_seed", 1.0)):
            with self.subTest(key=key, value=value):
                record = {**original, key: value}
                self.output.record_path.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.output.prepare()
        record = {**original, "job": {**original["job"], "slg_stop": True}}
        self.output.record_path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differ"):
            self.output.prepare()

    def test_wrong_dimensions_neither_publish_nor_skip(self):
        seed = self.output.prepare()
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.output.complete(Image.new("RGB", (32, 16)))
        self.assertFalse(self.output.is_complete())
        self.output.complete(Image.new("RGB", (16, 16)))
        Image.new("RGB", (32, 16)).save(self.output.image_path)
        self.assertFalse(self.output.is_complete())
        self.assertEqual(self.output.prepare(), seed)
        self.output.complete(Image.new("RGB", (16, 16)))
        self.assertTrue(self.output.is_complete())


if __name__ == "__main__":
    unittest.main()
