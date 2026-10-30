"""Regression checks for Mac adaptation; no weights or GPU required."""
import hashlib
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

import mps_runtime as runtime

MODEL = "OpenMOSS-Team/MOSS-Image-Base"


def fake_pipe(**attributes):
    transformer = SimpleNamespace(config=SimpleNamespace(num_layers=32), dtype=torch.bfloat16)
    return SimpleNamespace(transformer=transformer, remove_all_hooks=Mock(), **attributes)


def fake_output():
    return SimpleNamespace(images=[Image.new("RGB", (16, 16))])


class TimestepTests(unittest.TestCase):
    def test_cpu_patch_preserves_upstream_float64_math(self):
        source = (
            "import torch\nclass MOSSImagePipeline:\n"
            "    def embedding(self, times, half):\n"
            + runtime._FLOAT64_TIMESTEP + "        return embedding\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pipeline_moss_image.py").write_text(source)
            module = runtime._import_pipeline_module(root)
            times = torch.tensor([0.0, 0.125, 999.0], dtype=torch.float32)
            actual = module.MOSSImagePipeline().embedding(times, 128)
        phase = times.double()[:, None] * (10000 ** (-torch.arange(128, dtype=torch.float64) / 128))
        expected = torch.cat((phase.cos(), phase.sin()), -1).float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_changed_upstream_source_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pipeline_moss_image.py").write_text("# incompatible upstream pipeline\n")
            with self.assertRaisesRegex(ValueError, "changed; update"):
                runtime._import_pipeline_module(root)


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        events = self.events

        class BasePipeline:
            def encode_prompt(self, prompt, max_sequence_length):
                if self.text_encoder is None:
                    raise AssertionError("Encoding requires the text encoder")
                events.append("encode")
                weight = torch.ones(4, requires_grad=True)  # stands in for text-encoder parameters
                return torch.zeros(1, 2, 4) * weight, torch.ones(1, 2)

            def __call__(self, prompt, **kwargs):
                if self.transformer is None or self.text_encoder is not None:
                    raise AssertionError("Sampling must retain only the DiT")
                events.append("sample")
                return kwargs

        self.loader = Mock()
        module = SimpleNamespace(MOSSImagePipeline=BasePipeline, MOSSImageTransformer=self.loader)
        self.pipe = runtime._staged_pipeline_class(module)()
        self.pipe.root = Path("unused-model")
        self.pipe.text_encoder = None
        self.pipe.transformer = object()
        self.pipe.embeddings = OrderedDict()

        def load_text(*args, **kwargs):
            self.assertIsNone(self.pipe.transformer)
            self.events.append("load-text")
            return Mock()

        def load_dit(*args, **kwargs):
            self.assertIsNone(self.pipe.text_encoder)
            self.events.append("load-dit")
            return Mock()

        self.loader.from_pretrained.side_effect = load_dit
        self.text_patch = patch.object(runtime.AutoModel, "from_pretrained", side_effect=load_text)
        self.cache_patch = patch.object(runtime, "empty_cache")
        self.text_patch.start()
        self.cache_patch.start()
        self.addCleanup(self.text_patch.stop)
        self.addCleanup(self.cache_patch.stop)

    def test_both_prompts_encoded_before_sampling_and_cached(self):
        first = self.pipe("coffee", negative_prompt="blur")
        self.assertEqual(self.events, ["load-text", "encode", "encode", "load-dit", "sample"])
        self.events.clear()
        second = self.pipe("coffee", negative_prompt="blur", num_inference_steps=10, width=512)
        self.assertEqual(self.events, ["sample"])
        self.assertIs(first["prompt_embeds"], second["prompt_embeds"])

    def test_prompt_cache_evicts_least_recently_used(self):
        for index in range(runtime.CACHED_PROMPTS):
            self.pipe.encode_prompt(f"prompt-{index}")
        self.pipe.encode_prompt("prompt-0")
        self.pipe.encode_prompt("new")
        self.assertEqual(len(self.pipe.embeddings), runtime.CACHED_PROMPTS)
        self.assertIn((("prompt-0",), 4096), self.pipe.embeddings)
        self.assertNotIn((("prompt-1",), 4096), self.pipe.embeddings)

    def test_cached_embeddings_hold_no_autograd_graph(self):
        # A graph would keep the released text encoder's weights and activations alive.
        embeds, _ = self.pipe.encode_prompt("coffee")
        self.assertFalse(embeds.requires_grad)
        self.assertIsNone(embeds.grad_fn)

    def test_load_pipeline_records_the_bundle_pipeline_it_executes(self):
        class BasePipeline:
            def __init__(self, **components):
                self.text_encoder = None
                self.__dict__.update(components)

        module = SimpleNamespace(
            MOSSImagePipeline=BasePipeline, MOSSImageTransformer=Mock(), MOSSImageScheduler=Mock()
        )
        root = Path("bundle")
        with patch.object(runtime, "_bundle_root", return_value=root), \
             patch.object(runtime, "_import_pipeline_module", return_value=module), \
             patch.object(runtime.AutoTokenizer, "from_pretrained"), \
             patch.object(runtime.AutoencoderKLFlux2, "from_pretrained"):
            pipe = runtime.load_pipeline("model")
        self.assertEqual(pipe.pipeline_file, root / "pipeline_moss_image.py")


class RunnerTests(unittest.TestCase):
    def setUp(self):
        import gradio_app

        self.app = gradio_app
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.runner = gradio_app.ModelRunner(directory.name, local_files_only=True)
        self.config = gradio_app.GenerationConfig(prompt="coffee")
        for target, name in ((gradio_app, "set_scheduler"), (runtime, "empty_cache")):
            patcher = patch.object(target, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_failed_prompt_stage_releases_pipeline_and_reloads_it(self):
        broken, fresh = fake_pipe(), fake_pipe()

        def run(pipe, *args, **kwargs):
            if pipe is broken:
                pipe.transformer = None  # released for the text encoder, which then failed
                raise RuntimeError("MPS backend out of memory")
            return fake_output()

        with patch.object(self.app, "load_pipeline", side_effect=[broken, fresh]) as load, \
             patch.object(self.app, "generate_with_progress", side_effect=run):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                self.runner.generate(MODEL, self.config)
            self.assertIsNone(self.runner.pipe)
            broken.remove_all_hooks.assert_called_once()
            self.runner.generate(MODEL, self.config)
        self.assertEqual(load.call_count, 2)

    def test_mps_provenance_hashes_bundle_pipeline_and_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pipeline_moss_image.py"
            source.write_text("# bundle pipeline\n")
            with patch.object(self.app, "load_pipeline", return_value=fake_pipe(pipeline_file=source)), \
                 patch.object(self.app, "generate_with_progress", return_value=fake_output()):
                _, metadata, _ = self.runner.generate(MODEL, self.config)
        record = json.loads(Path(metadata).read_text())
        self.assertEqual(record["pipeline_sha256"], hashlib.sha256(b"# bundle pipeline\n").hexdigest())
        self.assertEqual(
            record["mps_runtime_sha256"], hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()
        )


class IntegrationTests(unittest.TestCase):
    def test_mac_routing_and_unsupported_compile(self):
        import gradio_app

        with patch.object(runtime, "is_available", return_value=True), patch.object(runtime, "load_pipeline") as load:
            gradio_app.load_pipeline("model", "model", local_files_only=True)
            load.assert_called_once_with("model", local_files_only=True)
            with self.assertRaisesRegex(ValueError, "Compile DiT"):
                gradio_app.load_pipeline("model", "model", compile_transformer=True)

    def test_cuda_routing_retains_existing_loader_and_configuration(self):
        import gradio_app

        with patch.object(runtime, "is_available", return_value=False), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(gradio_app, "load_hf_pipeline") as load, \
             patch.object(gradio_app, "configure_pipeline") as configure:
            result = gradio_app.load_pipeline("model", "sequential", local_files_only=True)
            load.assert_called_once_with("model", local_files_only=True)
            configure.assert_called_once_with(load.return_value, offload="sequential", compile_transformer=False)
            self.assertIs(result, configure.return_value)

    def test_gradio_ui_builds_for_mac_without_loading_weights(self):
        import gradio_app

        with patch.object(runtime, "is_available", return_value=True), patch.object(runtime, "load_pipeline") as load:
            app = gradio_app.build_app("missing-model-directory")
            self.assertEqual(app.title, "MOSS-Image")
            controls = {item["props"].get("label"): item["props"] for item in app.config["components"]}
            self.assertFalse(controls["Memory mode"]["interactive"])
            self.assertFalse(controls["Compile DiT"]["interactive"])
            load.assert_not_called()
            app.close()


if __name__ == "__main__":
    unittest.main()
