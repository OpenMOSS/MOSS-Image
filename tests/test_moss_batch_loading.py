"""CPU integration checks for component-selective loading with real Diffusers.

The tiny local pipeline models the official MOSS constructor contract. It tests
Diffusers loading/device behavior without downloading weights or claiming MOSS
image-quality parity; real-model parity is a separate validation.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from diffusers import DDPMScheduler, UNet2DModel
from transformers import BertTokenizer

from moss_batch.runtime import load_stage

PIPELINE = '''from diffusers import DiffusionPipeline, UNet2DModel, DDPMScheduler

class MOSSImageTransformer(UNet2DModel):
    pass

class MOSSImageScheduler(DDPMScheduler):
    pass

class MOSSImagePipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["text_encoder", "tokenizer", "vae"]

    def __init__(self, transformer: MOSSImageTransformer, scheduler: MOSSImageScheduler,
                 text_encoder=None, tokenizer=None, vae=None):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler,
                              text_encoder=text_encoder, tokenizer=tokenizer, vae=vae)
'''


class StageLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="moss-tiny-stage-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Diffusers copies all local custom pipelines into one shared cache name.
        # Keep the tiny fixture out of the user's real model-code cache.
        modules_cache = patch(
            "diffusers.utils.dynamic_modules_utils.HF_MODULES_CACHE",
            str(self.root / "module_cache"),
        )
        modules_cache.start()
        self.addCleanup(modules_cache.stop)
        (self.root / "pipeline_moss_image.py").write_text(PIPELINE, encoding="utf-8")
        index = {
            "_class_name": "MOSSImagePipeline", "_diffusers_version": "0.38.0",
            "transformer": ["pipeline_moss_image", "MOSSImageTransformer"],
            "scheduler": ["pipeline_moss_image", "MOSSImageScheduler"],
            "text_encoder": ["diffusers", "UNet2DModel"],
            "tokenizer": ["transformers", "BertTokenizer"],
            "vae": ["diffusers", "UNet2DModel"],
        }
        (self.root / "model_index.json").write_text(json.dumps(index), encoding="utf-8")

    def save_model(self, component):
        UNet2DModel(
            sample_size=8, in_channels=3, out_channels=3,
            down_block_types=("DownBlock2D",), up_block_types=("UpBlock2D",),
            block_out_channels=(8,), layers_per_block=1, norm_num_groups=4,
        ).save_pretrained(self.root / component)

    def test_encode_loads_without_any_generation_component_files(self):
        self.save_model("text_encoder")
        vocab = self.root / "vocab.txt"
        vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncat\n", encoding="utf-8")
        BertTokenizer(vocab_file=str(vocab)).save_pretrained(self.root / "tokenizer")
        pipe = load_stage(self.root, "encode", offload="none", device="cpu")
        self.assertIsNotNone(pipe.text_encoder)
        self.assertIsNotNone(pipe.tokenizer)
        self.assertIsNone(pipe.transformer)
        self.assertIsNone(pipe.scheduler)
        self.assertIsNone(pipe.vae)
        self.assertEqual(pipe._execution_device, torch.device("cpu"))
        self.assertEqual(pipe.text_encoder.dtype, torch.bfloat16)
        # Exercise the real hook sequence on CPU: absent components must be skipped.
        # Production CPU execution uses offload="none"; this only checks hook wiring.
        pipe.enable_model_cpu_offload(device="cpu")
        self.assertEqual(len(pipe._all_hooks), 1)
        self.assertEqual(pipe._execution_device.type, "cpu")
        pipe.remove_all_hooks()

    def test_generate_loads_without_encoder_or_tokenizer_files(self):
        self.save_model("transformer")
        self.save_model("vae")
        DDPMScheduler().save_pretrained(self.root / "scheduler")
        # Official HF bundles contain a custom module in each of these directories.
        # This exercises component-code trust separately from top-level pipeline trust.
        for component in ("transformer", "scheduler"):
            (self.root / component / "pipeline_moss_image.py").write_text(PIPELINE, encoding="utf-8")
        pipe = load_stage(self.root, "generate", offload="none", device="cpu")
        self.assertIsNone(pipe.text_encoder)
        self.assertIsNone(pipe.tokenizer)
        self.assertIsNotNone(pipe.transformer)
        self.assertIsNotNone(pipe.scheduler)
        self.assertIsNotNone(pipe.vae)
        self.assertEqual(type(pipe.transformer).__name__, "MOSSImageTransformer")
        self.assertEqual(type(pipe.scheduler).__name__, "MOSSImageScheduler")
        self.assertEqual(pipe._execution_device, torch.device("cpu"))
        self.assertEqual(pipe.transformer.dtype, torch.bfloat16)
        self.assertEqual(pipe.vae.dtype, torch.float32)
        pipe.enable_model_cpu_offload(device="cpu")
        self.assertEqual(len(pipe._all_hooks), 2)
        self.assertEqual(pipe._execution_device.type, "cpu")
        pipe.remove_all_hooks()


if __name__ == "__main__":
    unittest.main()
