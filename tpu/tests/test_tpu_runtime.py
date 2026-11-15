from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from collections import OrderedDict
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio_app  # noqa: E402
import inference_runtime  # noqa: E402
from tpu import tpu_runtime  # noqa: E402


def _sample_upstream_pipeline_source() -> str:
    return '''from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers import DiffusionPipeline


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _validate_skip_layers(layers, depth, name='skip_layers'):
    layers = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f'{name} must contain integer block indices in [0, {depth}).')
    return frozenset(layers)


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ffn_up = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim, dim, bias=False) for name in ('q', 'k', 'v', 'o')
        })
        self.proj['ffn_up_gate'] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj['ffn_down'] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj['q'].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj['o'](_attention(q, k, self.proj['v'](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj['ffn_up_gate'](y.to(dtype)).chunk(2, -1)
        return x + self.proj['ffn_down'](_situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer('bands', self._default_bands(torch.device('cpu')))

    def _default_bands(self, device):
        exponent = torch.arange(self.num_bands, device='cpu', dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.bands = self._default_bands(self.bands.device)

    def forward(self, height, width, text_length, device):
        # Diffusers/Accelerate can install tensors directly, bypassing load hooks.
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        # Coordinates: (r, f, h, w); sequence order is image followed by text.
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42,
                 text_dim=4096, freq_dim=256, latent_channels=128, rope_base=10000.0):
        super().__init__()
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def load_training_state_dict(self, state_dict, *, prefix=''):
        if prefix:
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
        return self.load_state_dict(state_dict, strict=True)

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=None, return_dict=True, skip_layers=None):
        skip_layers = _validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
            raise ValueError('hidden_states must be [B, latent_channels, H, W].')
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError('encoder_hidden_states must be [B, S, text_dim].')
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('encoder_attention_mask must be a binary [B, S] mask.')
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        # Training computes the sinusoid in float64 and feeds float32 into its MLP.
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index:index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError('Each sample needs at least one valid text token.')
            video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
            # Match the reference activation dtype under fp32 weights + autocast.
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width:])
            x = torch.cat((video, text), 1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    x = layer(x, rope, mod[index:index + 1])
            x = self.final_norm_video(x[:, :height * width].to(self.final_norm_video.weight.dtype))
            x = x * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        sample = torch.cat(outputs)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)


class MOSSImagePipeline(DiffusionPipeline):
    pass
'''


def _load_module_from_source(source: str, name: str):
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    return module


class _FakeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.devices: list[str] = []

    def to(self, *args, **kwargs):
        dev = args[0] if args else kwargs.get("device")
        if dev is not None:
            self.devices.append(str(dev))
        return super().to(*args, **kwargs)


class _FakeScheduler:
    def __init__(self, shift=5):
        self.config = {"shift": shift}
        self.timesteps = [4, 3, 2, 1]

    @classmethod
    def from_config(cls, config, *, shift):
        return cls(shift)


class _FakeBasePipeline:
    def __init__(self, *, include_text=True, image="fake-image"):
        self.text_encoder = _FakeModule() if include_text else None
        self.tokenizer = SimpleNamespace() if include_text else None
        self.transformer = _FakeModule()
        self.transformer.layers = nn.ModuleList([_FakeModule()])
        self.transformer.text_embedder = SimpleNamespace(blocks=nn.ModuleList([_FakeModule()]))
        self.vae = _FakeModule()
        self.scheduler = _FakeScheduler()
        self.components = {"transformer": self.transformer, "vae": self.vae}
        self.encode_calls: list[str] = []
        self.forward_calls: list[dict] = []
        self.image = image

    @property
    def _execution_device(self):
        return torch.device("cpu")

    def to(self, *args, **kwargs):
        return self

    def encode_prompt(self, prompt, max_sequence_length=4096):
        self.encode_calls.append(prompt)
        seq_len = max(2, len(prompt.split()) + 1)
        embeds = torch.arange(seq_len * 4, dtype=torch.float32).reshape(1, seq_len, 4)
        embeds.requires_grad_(True)
        mask = torch.ones((1, seq_len), dtype=torch.bool)
        return embeds, mask

    def __call__(self, **kwargs):
        self.forward_calls.append(kwargs)
        list(self.progress_bar(self.scheduler.timesteps))
        return SimpleNamespace(images=[self.image])

    def progress_bar(self, items):
        return items


def _load_tpu_cli():
    script_path = Path(__file__).resolve().parent.parent / "run_tpu.py"
    spec = importlib.util.spec_from_file_location("run_tpu", script_path)
    run_tpu = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(run_tpu)
    return run_tpu


class UpstreamPatchAndParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = _sample_upstream_pipeline_source()
        cls.upstream_mod = _load_module_from_source(raw, "test_upstream_moss_image")
        patched = tpu_runtime._patch_pipeline_source(raw)
        cls.patched_mod = _load_module_from_source(patched, "test_patched_tpu_moss_image")

    def _build_pair(self):
        torch.manual_seed(123)
        kwargs = dict(
            dim=64,
            ffn_dim=128,
            num_heads=4,
            num_layers=3,
            text_dim=32,
            freq_dim=64,
            latent_channels=16,
        )
        upstream_model = self.upstream_mod.MOSSImageTransformer(**kwargs).eval()
        patched_model = self.patched_mod.MOSSImageTransformer(**kwargs).eval()
        patched_model.load_training_state_dict(upstream_model.state_dict())
        return upstream_model, patched_model

    def test_changed_upstream_source_fails_explicitly(self):
        mutated = _sample_upstream_pipeline_source().replace(
            "mask[index].bool()", "mask[index].to(torch.bool)"
        )
        with self.assertRaises(ValueError):
            tpu_runtime._patch_pipeline_source(mutated)

    def test_static_padded_transformer_matches_unpadded_upstream_math(self):
        upstream_model, patched_model = self._build_pair()
        patched_model._static_attention_masking = True

        torch.manual_seed(7)
        hidden_states = torch.randn(2, 16, 4, 4, dtype=torch.float32)
        timestep = torch.tensor(412.5)
        raw_text = torch.randn(2, 10, 32, dtype=torch.float32)
        raw_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        # Pad out to bucket 16 with NaNs in masked-out positions to verify zero leakage.
        padded_text = torch.full((2, 16, 32), float("nan"), dtype=torch.float32)
        padded_text[:, :10] = raw_text
        padded_mask = torch.zeros((2, 16), dtype=torch.bool)
        padded_mask[:, :10] = raw_mask

        with torch.no_grad():
            for skip_layers in ((), (1,)):
                ref = upstream_model(
                    hidden_states,
                    timestep,
                    raw_text,
                    raw_mask,
                    skip_layers=skip_layers,
                ).sample
                out = patched_model(
                    hidden_states,
                    timestep,
                    padded_text,
                    padded_mask,
                    skip_layers=skip_layers,
                ).sample
                torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_non_contiguous_mask_matches_upstream_rope_positions(self):
        upstream_model, patched_model = self._build_pair()
        patched_model._static_attention_masking = True

        torch.manual_seed(9)
        hidden_states = torch.randn(1, 16, 4, 4, dtype=torch.float32)
        raw_text = torch.randn(1, 6, 32, dtype=torch.float32)
        mask = torch.tensor([[True, False, True, False, True, False]])
        with torch.no_grad():
            ref = upstream_model(hidden_states, 250.0, raw_text, mask).sample
            out = patched_model(hidden_states, 250.0, raw_text, mask).sample
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_tpu_float32_timestep_sinusoid_matches_float64_and_preserves_fp32_after_bf16(self):
        upstream_model, patched_model = self._build_pair()
        self.assertNotIn("_tpu_time_freq", patched_model.state_dict())

        patched_model.to(torch.bfloat16)
        self.assertEqual(patched_model._tpu_time_freq.dtype, torch.float32)
        self.assertEqual(patched_model.rope.bands.dtype, torch.float32)

        upstream_model, patched_model = self._build_pair()
        patched_model._force_tpu_timestep_sinusoid = True
        patched_model._static_attention_masking = True
        hidden_states = torch.randn(1, 16, 4, 4)
        enc = torch.randn(1, 8, 32)
        mask = torch.ones(1, 8, dtype=torch.bool)
        mask[:, 5:] = False
        with torch.no_grad():
            ref = upstream_model(hidden_states, 999.0, enc, mask).sample
            out = patched_model(hidden_states, 999.0, enc, mask).sample
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_fullgraph_compile_zero_recompilation_across_bucket_lengths(self):
        from torch._dynamo.testing import CompileCounterWithBackend

        upstream_model, patched_model = self._build_pair()
        counter = CompileCounterWithBackend("aot_eager")
        patched_model.compile(backend=counter, dynamic=False, fullgraph=True)

        torch.manual_seed(21)
        hidden = torch.randn(1, 16, 4, 4)
        t0 = torch.tensor(333.0)

        for true_len in (3, 9, 14):
            enc = torch.randn(1, 16, 32)
            mask = torch.zeros(1, 16, dtype=torch.bool)
            mask[:, :true_len] = True
            with torch.no_grad():
                ref = upstream_model(hidden, t0, enc[:, :true_len], torch.ones(1, true_len, dtype=torch.bool)).sample
                compiled = patched_model(hidden, t0, enc, mask).sample
            torch.testing.assert_close(compiled, ref, atol=1e-5, rtol=1e-5)
            self.assertEqual(counter.frame_count, 1)

    def test_static_sdpa_masks_expand_query_dimension(self):
        _, patched_model = self._build_pair()
        patched_model._static_attention_masking = True
        recorded_masks: list[tuple[int, int]] = []
        orig_sdpa = self.patched_mod.F.scaled_dot_product_attention

        def spy_sdpa(q, k, v, attn_mask=None, **kwargs):
            if attn_mask is not None:
                recorded_masks.append((q.shape[-2], attn_mask.shape[-2]))
                self.assertEqual(attn_mask.shape[-2], q.shape[-2])
            return orig_sdpa(q, k, v, attn_mask=attn_mask, **kwargs)

        hidden = torch.randn(1, 16, 4, 4)
        enc = torch.randn(1, 12, 32)
        mask = torch.zeros(1, 12, dtype=torch.bool)
        mask[:, :5] = True

        with (
            mock.patch.object(self.patched_mod.F, "scaled_dot_product_attention", side_effect=spy_sdpa),
            torch.no_grad(),
        ):
            patched_model(hidden, 200.0, enc, mask)
            # Also verify direct _attention caller with singleton query mask gets expanded.
            q = torch.randn(1, 7, 64)
            k = torch.randn(1, 7, 64)
            v = torch.randn(1, 7, 64)
            rope = torch.ones(1, 7, 16)
            singleton_mask = torch.ones(1, 1, 1, 7, dtype=torch.bool)
            self.patched_mod._attention(q, k, v, rope, 4, attn_mask=singleton_mask)

        # 2 text blocks (Q=12) + 3 main blocks (Q=4*4+12=28) + 1 direct call (Q=7)
        self.assertEqual(recorded_masks, [(12, 12), (12, 12), (28, 28), (28, 28), (28, 28), (7, 7)])


class BucketingAndStagingTests(unittest.TestCase):
    def test_select_prompt_bucket_and_static_pad(self):
        self.assertEqual(tpu_runtime.select_prompt_bucket(1), 512)
        self.assertEqual(tpu_runtime.select_prompt_bucket(512), 512)
        self.assertEqual(tpu_runtime.select_prompt_bucket(513), 1024)
        self.assertEqual(tpu_runtime.select_prompt_bucket(4096), 4096)
        with self.assertRaises(ValueError):
            tpu_runtime.select_prompt_bucket(0)
        with self.assertRaises(ValueError):
            tpu_runtime.select_prompt_bucket(4097)

        emb = torch.ones(1, 7, 16)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
        padded_emb, padded_mask = tpu_runtime.static_pad_embeds(emb, mask, buckets=(8, 16))
        self.assertEqual(tuple(padded_emb.shape), (1, 8, 16))
        self.assertEqual(padded_mask.dtype, torch.bool)
        self.assertEqual(int(padded_mask.sum().item()), 5)

        with self.assertRaises(ValueError):
            tpu_runtime.static_pad_embeds(emb, torch.tensor([[2, 1, 1, 1, 1, 0, 0]]))
        with self.assertRaises(ValueError):
            tpu_runtime.static_pad_embeds(emb, torch.zeros((1, 7), dtype=torch.bool))

    def test_staged_pipeline_encodes_caches_and_pads_to_static_bucket(self):
        base = _FakeBasePipeline()
        with (
            mock.patch.object(tpu_runtime, "empty_cache") as mock_empty,
            mock.patch.object(tpu_runtime, "configure_eager_mode"),
            mock.patch.object(tpu_runtime, "configure_precision"),
        ):
            pipe = tpu_runtime.StagedTPUImagePipeline(
                base,
                pipeline_file=Path(__file__),
                device=torch.device("cpu"),
                offload="model",
                prompt_buckets=(8, 16),
            )
            self.assertEqual(pipe._stage, "cpu")
            result = pipe("two words", negative_prompt="bad", num_inference_steps=2, guidance_scale=2.0)
            self.assertEqual(result.images, ["fake-image"])
            self.assertEqual(base.encode_calls, ["two words", "bad"])
            call = base.forward_calls[-1]
            self.assertEqual(tuple(call["prompt_embeds"].shape), (1, 8, 4))
            self.assertEqual(int(call["prompt_attention_mask"].sum().item()), 3)
            self.assertEqual(tuple(call["negative_prompt_embeds"].shape), (1, 8, 4))
            self.assertFalse(call["prompt_embeds"].requires_grad)
            # Second call with same prompts must reuse cache without invoking Stage 1.
            base.encode_calls.clear()
            pipe("two words", negative_prompt="bad", num_inference_steps=2, guidance_scale=2.0)
            self.assertEqual(base.encode_calls, [])
            self.assertEqual(mock_empty.call_count, 1)

    def test_prompt_cache_evicts_lru_and_decoupled_precomputed_skips_text_encoder(self):
        base = _FakeBasePipeline(include_text=False)
        with (
            mock.patch.object(tpu_runtime, "empty_cache"),
            mock.patch.object(tpu_runtime, "configure_eager_mode"),
            mock.patch.object(tpu_runtime, "configure_precision"),
        ):
            pipe = tpu_runtime.StagedTPUImagePipeline(
                base,
                pipeline_file=Path(__file__),
                device=torch.device("cpu"),
                offload="none",
                prompt_buckets=(16, 32),
            )
            with self.assertRaises(ValueError):
                pipe("requires text encoder")

            pre_pos = torch.randn(1, 5, 4)
            pre_neg = torch.randn(1, 3, 4)
            pipe(prompt_embeds=pre_pos, negative_prompt_embeds=pre_neg, guidance_scale=3.0)
            call = base.forward_calls[-1]
            self.assertEqual(tuple(call["prompt_embeds"].shape), (1, 16, 4))
            self.assertEqual(int(call["prompt_attention_mask"].sum().item()), 5)
            self.assertEqual(tuple(call["negative_prompt_embeds"].shape), (1, 16, 4))

        # Test LRU eviction count
        base_with_text = _FakeBasePipeline(include_text=True)
        with (
            mock.patch.object(tpu_runtime, "empty_cache"),
            mock.patch.object(tpu_runtime, "configure_eager_mode"),
            mock.patch.object(tpu_runtime, "configure_precision"),
        ):
            staged = tpu_runtime.StagedTPUImagePipeline(
                base_with_text,
                pipeline_file=Path(__file__),
                device=torch.device("cpu"),
                prompt_buckets=(8,),
            )
            for idx in range(tpu_runtime.CACHED_PROMPTS + 1):
                staged.encode_prompt(f"prompt {idx}")
            self.assertEqual(len(staged._prompt_cache), tpu_runtime.CACHED_PROMPTS)
            self.assertIsInstance(staged._prompt_cache, OrderedDict)
            self.assertNotIn(("prompt 0", 4096), staged._prompt_cache)
            self.assertEqual(staged._stage, "cpu")
            base_with_text.encode_calls.clear()
            prompts = ["new prompt", "prompt 1", "new prompt"]
            embeds, mask = staged.encode_prompt(prompts)
            self.assertEqual(base_with_text.encode_calls, ["new prompt"])
            self.assertEqual(embeds.shape[0], len(prompts))
            self.assertEqual(mask.sum(dim=1).tolist(), [3, 3, 3])
            self.assertEqual(staged.encode_prompt([f"new {i}" for i in range(9)])[0].shape[0], 9)
            self.assertEqual(len(staged._prompt_cache), tpu_runtime.CACHED_PROMPTS)


class IntegrationTests(unittest.TestCase):
    def test_tpu_availability_detection(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertFalse(tpu_runtime.is_available())

        fake_accel = SimpleNamespace(
            is_available=lambda: True,
            current_accelerator=lambda: SimpleNamespace(type="tpu"),
        )
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch, "accelerator", fake_accel, create=True),
        ):
            self.assertTrue(tpu_runtime.is_available())

    def test_gradio_routing_and_provenance_for_tpu(self):
        sentinel_pipe = SimpleNamespace()
        with (
            mock.patch.object(gradio_app.mps_runtime, "is_available", return_value=False),
            mock.patch.object(tpu_runtime, "is_available", return_value=True),
            mock.patch.object(tpu_runtime, "load_pipeline", return_value=sentinel_pipe) as mock_load,
        ):
            loaded = gradio_app.load_pipeline(
                "some-model", offload="model", compile_transformer=True, local_files_only=True
            )
            self.assertIs(loaded, sentinel_pipe)
            mock_load.assert_called_once_with(
                "some-model", offload="model", compile_transformer=True, local_files_only=True
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "bundle"
            (model_dir / "tokenizer").mkdir(parents=True)
            bundle_pipe = model_dir / "pipeline_moss_image.py"
            bundle_bytes = b"# bundle pipeline\n"
            bundle_pipe.write_bytes(bundle_bytes)
            out_dir = root / "outputs"

            base = _FakeBasePipeline(image=Image.new("RGB", (16, 16)))
            base.transformer.config = SimpleNamespace(num_layers=42)
            base.transformer.dtype = torch.bfloat16
            base.remove_all_hooks = mock.MagicMock()

            with (
                mock.patch.object(tpu_runtime, "empty_cache") as mock_empty_cache,
                mock.patch.object(tpu_runtime, "configure_eager_mode"),
                mock.patch.object(tpu_runtime, "configure_precision"),
            ):
                staged_pipe = tpu_runtime.StagedTPUImagePipeline(
                    base,
                    pipeline_file=bundle_pipe,
                    device=torch.device("cpu"),
                    offload="model",
                    prompt_buckets=(16,),
                )

                runner = gradio_app.ModelRunner(output_dir=out_dir, local_files_only=True)
                cfg = gradio_app.GenerationConfig(
                    prompt="cat",
                    steps=4,
                    seed=42,
                    offload="model",
                    compile_transformer=True,
                )
                with (
                    mock.patch.object(gradio_app, "model_source", return_value=str(model_dir)),
                    mock.patch.object(gradio_app, "is_model_bundle", return_value=True),
                    mock.patch.object(gradio_app, "load_pipeline", return_value=staged_pipe),
                    mock.patch.object(torch.cuda, "is_available", return_value=False),
                    mock.patch.object(gradio_app.mps_runtime, "is_available", return_value=False),
                    mock.patch.object(tpu_runtime, "is_available", return_value=True),
                    mock.patch.object(tpu_runtime, "device_name", return_value="TPU v6e-1"),
                ):
                    progress = mock.Mock()
                    _paths, meta_path, _status = runner.generate(str(model_dir), cfg, progress)
                    rec = json.loads(Path(meta_path).read_text(encoding="utf-8"))
                    runner.unload()

                base.remove_all_hooks.assert_called_once()
                self.assertGreaterEqual(mock_empty_cache.call_count, 1)
                progress.assert_any_call((4, 4), desc="Sampling · step 4/4", unit="steps")
                self.assertEqual(base.scheduler.config["shift"], cfg.shift)

            self.assertEqual(rec["device"], "tpu")
            self.assertEqual(rec["gpu"], "TPU v6e-1")
            self.assertEqual(rec["pipeline_sha256"], hashlib.sha256(bundle_bytes).hexdigest())
            self.assertEqual(
                rec["tpu_runtime_sha256"],
                hashlib.sha256(Path(tpu_runtime.__file__).read_bytes()).hexdigest(),
            )

    def test_configure_pipeline_routes_to_tpu_compile(self):
        fake_pipe = _FakeBasePipeline()
        fake_pipe.transformer.compile = mock.MagicMock()
        pipe = tpu_runtime.StagedTPUImagePipeline(
            fake_pipe, __file__, device=torch.device("cpu"), offload="none"
        )
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(tpu_runtime, "is_available", return_value=True),
            mock.patch.object(tpu_runtime, "tpu_budget_gib", return_value=32.0),
            mock.patch.object(tpu_runtime, "load_pipeline", return_value=pipe) as load,
        ):
            loaded = inference_runtime.load_pipeline(inference_runtime.DEFAULT_MODEL, local_files_only=True)
            load.assert_called_once_with(inference_runtime.DEFAULT_MODEL, local_files_only=True)
            configured = inference_runtime.configure_pipeline(loaded, offload="auto", compile_transformer=True)
            self.assertIs(configured, pipe)
            self.assertEqual(pipe._stage, "diffusion")
            self.assertEqual(pipe._offload, "model")
            fake_pipe.transformer.compile.assert_called_once_with(
                backend="tpu", dynamic=False, fullgraph=True
            )
            inference_runtime.configure_pipeline(pipe, offload="none", compile_transformer=True)
            self.assertEqual(pipe._stage, "all")
            self.assertEqual(fake_pipe.transformer.compile.call_count, 1)

    def test_run_tpu_probe_cli_allow_cpu(self):
        run_tpu = _load_tpu_cli()
        parser = run_tpu.build_parser()
        args = parser.parse_args(["--mode", "probe", "--allow-cpu"])
        with mock.patch("builtins.print"):
            info = run_tpu.run_probe(args)
        self.assertIn("torch_version", info)
        self.assertIn("matmul_smoke_ms", info)

    def test_probe_exits_unsuccessfully_without_a_tpu(self):
        with mock.patch.object(tpu_runtime, "is_available", return_value=False), mock.patch("builtins.print"):
            self.assertEqual(_load_tpu_cli().main(["--mode", "probe"]), 1)

    def test_benchmark_times_actual_denoising_and_preserves_outputs(self):
        cli = _load_tpu_cli()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sample.png"
            args = cli.build_parser().parse_args([
                "--mode", "benchmark", "--allow-cpu", "--steps", "5",
                "--bench-modes", "eager,compile", "--output", str(output),
            ])
            base = _FakeBasePipeline(image=Image.new("RGB", (16, 16)))
            pipe = tpu_runtime.StagedTPUImagePipeline(base, __file__, device=torch.device("cpu"))
            progress_bar = pipe.progress_bar
            with (
                mock.patch.object(tpu_runtime, "is_available", return_value=False),
                mock.patch.object(tpu_runtime, "load_pipeline", return_value=pipe),
                mock.patch.object(cli.time, "perf_counter", side_effect=[0, 1, 3, 8] * 4),
                mock.patch("builtins.print"),
            ):
                report = cli.run_generate_or_benchmark(args)
            paths = set()
            for mode, runs in report["results"].items():
                for idx, run in enumerate(runs, start=1):
                    self.assertEqual(run["elapsed_seconds"], 8)
                    self.assertEqual(run["denoising_steps"], 4)
                    self.assertEqual(run["denoising_seconds"], 2)
                    self.assertEqual(run["ms_per_step"], 500)
                    path = Path(run["output_image"])
                    self.assertEqual(path.name, f"sample-{mode}-{idx}.png")
                    self.assertTrue(path.is_file())
                    paths.add(path)
            self.assertEqual(len(paths), 4)
            self.assertEqual(pipe.progress_bar, progress_bar)

    def test_hub_snapshot_resolution_for_split_stages(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            tpu_runtime, "snapshot_download", return_value=tmp
        ) as download:
            cli = _load_tpu_cli()
            args = cli.build_parser().parse_args(["--mode", "encode-only", "--local-files-only", "--allow-cpu"])
            with mock.patch.object(tpu_runtime, "is_available", return_value=False), mock.patch.object(
                tpu_runtime, "_import_pipeline_module", side_effect=RuntimeError("stop before loading weights")
            ) as importer:
                with self.assertRaisesRegex(RuntimeError, "stop before loading weights"):
                    cli.run_encode_only(args)
                importer.assert_called_once_with(Path(tmp))
                self.assertEqual(download.call_args.args, (inference_runtime.DEFAULT_MODEL,))
                self.assertTrue(download.call_args.kwargs["local_files_only"])
                self.assertIn("text_encoder/*", download.call_args.kwargs["allow_patterns"])
                self.assertNotIn("transformer/*", download.call_args.kwargs["allow_patterns"])

                with self.assertRaisesRegex(RuntimeError, "stop before loading weights"):
                    tpu_runtime.load_pipeline(inference_runtime.DEFAULT_MODEL, include_text_encoder=False)
                self.assertIn("transformer/*", download.call_args.kwargs["allow_patterns"])
                self.assertNotIn("text_encoder/*", download.call_args.kwargs["allow_patterns"])
                (Path(tmp) / "pipeline_moss_image.py").touch()
                self.assertEqual(tpu_runtime.resolve_model_bundle(tmp), Path(tmp).resolve())
                self.assertEqual(download.call_count, 2)

    def test_eager_mode_uses_torchtpu_context(self):
        class EagerMode(Enum):
            DEFER_NEVER = 0
            DEFER_AND_FUSE = 1

        tpu = SimpleNamespace(EagerMode=EagerMode, eager_mode=EagerMode.DEFER_NEVER)

        @contextmanager
        def set_eager_mode(mode):
            previous = tpu.eager_mode
            tpu.eager_mode = mode
            try:
                yield
            finally:
                tpu.eager_mode = previous

        tpu.set_eager_mode = set_eager_mode
        with mock.patch.object(torch, "tpu", tpu, create=True):
            device = SimpleNamespace(type="tpu")
            with tpu_runtime.configure_eager_mode("defer_and_fuse", device):
                self.assertEqual(tpu.eager_mode, EagerMode.DEFER_AND_FUSE)
                with tpu_runtime.configure_eager_mode("default", device):
                    self.assertEqual(tpu.eager_mode, EagerMode.DEFER_NEVER)
                self.assertEqual(tpu.eager_mode, EagerMode.DEFER_AND_FUSE)
            self.assertEqual(tpu.eager_mode, EagerMode.DEFER_NEVER)

    def test_ensure_group_norm_contiguous_normalizes_channels_last_inputs(self):
        observed_contiguous = []

        class _CheckingGroupNorm(torch.nn.GroupNorm):
            def forward(self, input: torch.Tensor) -> torch.Tensor:
                observed_contiguous.append(input.is_contiguous())
                return super().forward(input)

        class _FakeVae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = _CheckingGroupNorm(32, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.norm(x)

        vae = _FakeVae()
        self.assertEqual(tpu_runtime.ensure_group_norm_contiguous(vae), 1)
        self.assertEqual(tpu_runtime.ensure_group_norm_contiguous(vae), 0)

        x_nchw = torch.randn(1, 64, 8, 8)
        x_nhwc = x_nchw.to(memory_format=torch.channels_last)
        self.assertFalse(x_nhwc.is_contiguous())

        out_positional = vae(x_nhwc)
        out_kwarg = vae.norm(input=x_nhwc)
        ref = torch.nn.functional.group_norm(
            x_nchw, 32, vae.norm.weight, vae.norm.bias, vae.norm.eps
        )
        self.assertEqual(observed_contiguous, [True, True])
        torch.testing.assert_close(out_positional, ref)
        torch.testing.assert_close(out_kwarg, ref)

        base = _FakeBasePipeline()
        base.vae = _FakeVae()
        pipe = tpu_runtime.StagedTPUImagePipeline(base, __file__, device=torch.device("cpu"))
        observed_contiguous.clear()
        pipe.vae(x_nhwc)
        self.assertEqual(observed_contiguous, [True])

    def test_run_tpu_cli_defaults_disable_experimental_compile(self):
        parser = _load_tpu_cli().build_parser()
        defaults = parser.parse_args([])
        self.assertFalse(defaults.compile)
        self.assertEqual(defaults.bench_modes, "eager,defer_and_fuse")
        self.assertEqual(defaults.stage_switch_mode, "auto")
        self.assertEqual(defaults.switch_modes, "d2h,drop_tpu")
        explicit_compile = parser.parse_args(["--compile", "--stage-switch-mode", "drop_tpu"])
        self.assertTrue(explicit_compile.compile)
        self.assertEqual(explicit_compile.stage_switch_mode, "drop_tpu")

    def test_stage_switch_mode_resolution_and_meta_drop_restore_parity(self):
        self.assertEqual(tpu_runtime.resolve_stage_switch_mode("auto", torch.device("cpu")), "d2h")
        self.assertEqual(
            tpu_runtime.resolve_stage_switch_mode("auto", torch.device("meta"), host_total_gib=64.0),
            "drop_tpu",
        )
        self.assertEqual(
            tpu_runtime.resolve_stage_switch_mode("auto", torch.device("meta"), host_total_gib=24.0),
            "d2h",
        )
        with self.assertRaises(ValueError):
            tpu_runtime.validate_stage_switch_mode("invalid")

        raw = _sample_upstream_pipeline_source()
        patched_mod = _load_module_from_source(tpu_runtime._patch_pipeline_source(raw), "test_drop_restore_mod")
        torch.manual_seed(11)
        model = patched_mod.MOSSImageTransformer(
            dim=64, ffn_dim=128, num_heads=4, num_layers=2, text_dim=32, freq_dim=64, latent_channels=16
        ).eval()
        # Tie two parameters to ensure snapshot/restore preserves parameter object sharing.
        model.layers[1].norm1.weight = model.layers[0].norm1.weight
        model.to(torch.bfloat16)
        model._force_tpu_timestep_sinusoid = True
        model._static_attention_masking = True

        x = torch.randn(1, 16, 4, 4, dtype=torch.bfloat16)
        enc = torch.randn(1, 8, 32, dtype=torch.bfloat16)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
        with torch.no_grad():
            ref = model(x, 300.0, enc, mask).sample.clone()

        snap = tpu_runtime._snapshot_module_cpu_weights(model)
        tpu_runtime._release_module_to_meta(model)
        self.assertTrue(all(p.is_meta for p in model.parameters()))
        self.assertTrue(model._tpu_time_freq.is_meta)
        self.assertTrue(model.rope.bands.is_meta)
        # Retaining snapshots must preserve exact FP32 CPU buffers untouched.
        self.assertFalse(any(entry[-1].is_meta for entry in snap["params"]))
        self.assertFalse(any(entry[-1].is_meta for entry in snap["buffers"]))

        tpu_runtime._restore_module_from_cpu_snapshot(model, snap, torch.device("cpu"))
        self.assertFalse(any(p.is_meta for p in model.parameters()))
        self.assertIs(model.layers[1].norm1.weight, model.layers[0].norm1.weight)
        self.assertEqual(model._tpu_time_freq.dtype, torch.float32)
        self.assertEqual(model.rope.bands.dtype, torch.float32)
        with torch.no_grad():
            restored = model(x, 300.0, enc, mask).sample
        torch.testing.assert_close(restored, ref)

    def test_staged_pipeline_drop_tpu_skips_d2h_and_clears_snapshots_for_offload_none(self):
        base = _FakeBasePipeline()
        with (
            mock.patch.object(tpu_runtime, "empty_cache"),
            mock.patch.object(tpu_runtime, "synchronize"),
            mock.patch.object(tpu_runtime, "configure_eager_mode"),
            mock.patch.object(tpu_runtime, "configure_precision"),
        ):
            pipe = tpu_runtime.StagedTPUImagePipeline(
                base,
                pipeline_file=Path(__file__),
                device=torch.device("cpu"),
                offload="model",
                stage_switch_mode="drop_tpu",
                prompt_buckets=(8,),
            )
            pipe._force_drop_tpu_for_test = True
            self.assertEqual(pipe.effective_stage_switch_mode, "drop_tpu")

            # Cold encode_prompt from cpu must not load vae.
            pipe.reset_stage_switch_history()
            pipe.encode_prompt("encode only prompt")
            encode_history = pipe.stage_switch_history
            self.assertEqual([(h["from_stage"], h["to_stage"]) for h in encode_history], [
                ("cpu", "text"),
                ("text", "cpu"),
            ])
            self.assertEqual(encode_history[0]["load_modules"], ["text_encoder"])
            self.assertEqual(base.vae.devices, [])

            # Prompt 1: cpu -> text -> diffusion (vae loads once in text -> diffusion, zero offload .to('cpu')).
            pipe.reset_stage_switch_history()
            pipe("first prompt", num_inference_steps=2, guidance_scale=1.0)
            p1_history = pipe.stage_switch_history
            self.assertEqual(p1_history[0]["load_modules"], ["text_encoder"])
            self.assertEqual(p1_history[1]["load_modules"], ["transformer", "vae"])
            self.assertEqual(pipe._stage, "diffusion")
            self.assertEqual(base.text_encoder.devices, ["cpu"])
            self.assertEqual(base.transformer.devices, ["cpu"])
            self.assertEqual(base.vae.devices, ["cpu"])
            self.assertTrue(base.text_encoder.weight.is_meta)
            self.assertIn("text_encoder", pipe._meta_offloaded_modules)
            self.assertIn("transformer", pipe._cpu_weight_snapshots)

            base.text_encoder.devices.clear()
            base.transformer.devices.clear()
            base.vae.devices.clear()

            # Interactive Prompt 2: diffusion -> text -> diffusion keeps vae resident across stages.
            pipe.reset_stage_switch_history()
            pipe("second prompt", num_inference_steps=2, guidance_scale=1.0)
            history = pipe.stage_switch_history
            self.assertEqual([(h["from_stage"], h["to_stage"]) for h in history], [
                ("diffusion", "text"),
                ("text", "diffusion"),
            ])
            self.assertEqual(history[0]["offload_modules"], ["transformer"])
            self.assertEqual(history[0]["load_modules"], ["text_encoder"])
            self.assertEqual(history[1]["offload_modules"], ["text_encoder"])
            self.assertEqual(history[1]["load_modules"], ["transformer"])
            self.assertTrue(all(h["d2h_seconds"] == 0.0 for h in history))
            self.assertEqual(base.text_encoder.devices, [])
            self.assertEqual(base.transformer.devices, [])
            self.assertEqual(base.vae.devices, [])
            self.assertTrue(base.text_encoder.weight.is_meta)
            self.assertFalse(base.transformer.weight.is_meta)
            self.assertFalse(base.vae.weight.is_meta)

            # Explicit cpu offload still evicts resident vae.
            pipe.reset_stage_switch_history()
            pipe._activate_stage("cpu")
            cpu_history = pipe.stage_switch_history
            self.assertEqual(cpu_history[-1]["offload_modules"], ["transformer", "vae"])
            self.assertTrue(base.vae.weight.is_meta)

            # Replacing vae while staying on diffusion reloads the new module on the next forward pass.
            pipe("third prompt", num_inference_steps=2, guidance_scale=1.0)
            replacement_vae = _FakeModule()
            pipe.vae = replacement_vae
            pipe.reset_stage_switch_history()
            pipe("third prompt", num_inference_steps=2, guidance_scale=1.0)
            self.assertEqual(pipe.stage_switch_history[-1]["load_modules"], ["vae"])
            self.assertEqual(replacement_vae.devices, ["cpu"])

            # Switching back to d2h restores parked meta text_encoder onto CPU and frees snapshots.
            pipe.set_stage_switch_mode("d2h")
            self.assertFalse(base.text_encoder.weight.is_meta)
            self.assertEqual(pipe._meta_offloaded_modules, set())
            self.assertEqual(pipe._cpu_weight_snapshots, {})

            # d2h prompt switch also keeps vae resident between diffusion and text stages.
            pipe.reset_stage_switch_history()
            pipe("fourth prompt", num_inference_steps=2, guidance_scale=1.0)
            d2h_history = pipe.stage_switch_history
            self.assertEqual(d2h_history[0]["offload_modules"], ["transformer"])
            self.assertEqual(d2h_history[1]["load_modules"], ["transformer"])

            # offload="none" must not keep duplicate CPU weight snapshots.
            pipe.set_stage_switch_mode("drop_tpu")
            pipe.configure_offload("none")
            self.assertEqual(pipe._stage, "all")
            self.assertEqual(pipe._cpu_weight_snapshots, {})

    def test_run_tpu_stage_switch_bench_allow_cpu(self):
        cli = _load_tpu_cli()
        with tempfile.TemporaryDirectory() as tmp:
            out_img = Path(tmp) / "switch.png"
            report_file = Path(tmp) / "switch_report.json"
            base = _FakeBasePipeline(image=Image.new("RGB", (8, 8), color=(12, 34, 56)))
            pipe = tpu_runtime.StagedTPUImagePipeline(base, __file__, device=torch.device("cpu"))
            args = cli.build_parser().parse_args(
                [
                    "--mode",
                    "stage-switch-bench",
                    "--allow-cpu",
                    "--steps",
                    "2",
                    "--output",
                    str(out_img),
                    "--save-report",
                    str(report_file),
                ]
            )
            with (
                mock.patch.object(tpu_runtime, "is_available", return_value=False),
                mock.patch.object(tpu_runtime, "load_pipeline", return_value=pipe),
                mock.patch("builtins.print"),
            ):
                report = cli.run_stage_switch_bench(args)

            self.assertTrue(report_file.is_file())
            self.assertEqual(report["mode"], "stage-switch-bench")
            self.assertIn("d2h", report["results"])
            self.assertIn("drop_tpu", report["results"])
            self.assertIn("comparison", report)
            self.assertEqual(
                [
                    (h["from_stage"], h["to_stage"])
                    for h in report["results"]["drop_tpu"]["run_2_interactive_prompt_switch"]["stage_switches"]
                ],
                [("diffusion", "text"), ("text", "diffusion")],
            )
            self.assertEqual(
                report["results"]["drop_tpu"]["parity_vs_baseline"]["run_2"]["mean_abs_diff"], 0.0
            )
            self.assertEqual(
                report["results"]["drop_tpu"]["parity_vs_baseline"]["run_2"]["cosine_similarity"], 1.0
            )

            # High-res uint8 RGB images must also retain exact cosine_similarity == 1.0 in float64.
            hi_res = Image.effect_noise((1200, 848), 25.0).convert("RGB")
            hi_res_stats = cli._image_parity_stats(hi_res, hi_res)
            self.assertEqual(hi_res_stats["cosine_similarity"], 1.0)
            self.assertEqual(hi_res_stats["max_abs_diff"], 0.0)

    def test_restore_snapshot_uses_non_blocking_for_tpu_device(self):
        from types import SimpleNamespace

        mod = torch.nn.Linear(4, 4, bias=False)
        snap = tpu_runtime._snapshot_module_cpu_weights(mod)
        calls: list[tuple[str, bool]] = []
        orig_to = torch.Tensor.to

        def spy_to(tensor_self, *args, **kwargs):
            dev_arg = args[0] if args else kwargs.get("device")
            nb_arg = bool(kwargs.get("non_blocking", False))
            if getattr(dev_arg, "type", None) == "tpu":
                calls.append((dev_arg.type, nb_arg))
                return tensor_self.clone()
            return orig_to(tensor_self, *args, **kwargs)

        tpu_runtime._release_module_to_meta(mod)
        fake_tpu_dev = SimpleNamespace(type="tpu")
        with mock.patch.object(torch.Tensor, "to", new=spy_to):
            tpu_runtime._restore_module_from_cpu_snapshot(mod, snap, fake_tpu_dev)
        self.assertEqual(calls, [("tpu", True)])

    def test_gradio_ui_disables_compile_dit_on_tpu(self):
        controls = {}
        change_handlers = []

        class _FakeWidget:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                label = kwargs.get("label")
                if label is not None:
                    controls[label] = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def _record_event(self, fn=None, *_args, **_kwargs):
                if callable(fn):
                    change_handlers.append(fn)

            change = _record_event
            click = _record_event
            input = _record_event
            release = _record_event
            select = _record_event
            tick = _record_event

        class _FakeBlocks(_FakeWidget):
            def serve_static_file(self, path):
                return {"url": f"/file={Path(path).name}"}

            def queue(self, **_kwargs):
                return self

        fake_gradio = SimpleNamespace(
            Blocks=_FakeBlocks,
            Row=_FakeWidget,
            Column=_FakeWidget,
            Group=_FakeWidget,
            Accordion=_FakeWidget,
            Dropdown=_FakeWidget,
            Textbox=_FakeWidget,
            State=_FakeWidget,
            Slider=_FakeWidget,
            Number=_FakeWidget,
            Radio=_FakeWidget,
            Checkbox=_FakeWidget,
            Button=_FakeWidget,
            Gallery=_FakeWidget,
            Markdown=_FakeWidget,
            HTML=_FakeWidget,
            Timer=_FakeWidget,
            Progress=lambda: None,
            SelectData=object,
            Error=RuntimeError,
            update=lambda **kwargs: kwargs,
            skip=lambda: None,
        )

        with (
            mock.patch.dict(sys.modules, {"gradio": fake_gradio}),
            mock.patch.object(gradio_app.mps_runtime, "is_available", return_value=False),
            mock.patch.object(tpu_runtime, "is_available", return_value=True),
        ):
            gradio_app.build_app("missing-model-directory")

        self.assertTrue(controls["Memory mode"]["interactive"])
        self.assertFalse(controls["Compile DiT"]["interactive"])
        self.assertFalse(controls["Compile DiT"]["value"])
        update_fns = [fn for fn in change_handlers if getattr(fn, "__name__", "") == "update_compile"]
        self.assertEqual(len(update_fns), 1)
        self.assertEqual(update_fns[0]("none"), {"value": False, "interactive": False})


if __name__ == "__main__":
    unittest.main()
