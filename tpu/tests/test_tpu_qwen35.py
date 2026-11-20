from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpu import tpu_qwen35, tpu_runtime  # noqa: E402


def _load_script_module(rel_path: str, name: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _naive_causal_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, None]:
    """Minimal pure-PyTorch causal Gated DeltaNet recurrence reference for CPU tests."""
    b, seq_len, hk, kdim = query.shape
    hv = value.shape[2]
    vdim = value.shape[-1]
    if hk != hv:
        rep = hv // hk
        query = query.repeat_interleave(rep, dim=2)
        key = key.repeat_interleave(rep, dim=2)

    q = query.float()
    k = key.float()
    v = value.float()
    if use_qk_l2norm_in_kernel:
        q = q / (q.pow(2).sum(dim=-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.pow(2).sum(dim=-1, keepdim=True) + 1e-6).sqrt()
    q = q * (kdim**-0.5)

    state = torch.zeros((b, hv, kdim, vdim), dtype=torch.float32, device=query.device)
    outputs = []
    for t in range(seq_len):
        decay = g[:, t, :, None, None].float().exp()
        state = state * decay
        k_t = k[:, t]  # (B, H, K)
        v_t = v[:, t]  # (B, H, V)
        beta_t = beta[:, t, :, None].float()
        kv_mem = torch.einsum("bhk,bhkv->bhv", k_t, state)
        delta = (v_t - kv_mem) * beta_t
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        out_t = torch.einsum("bhk,bhkv->bhv", q[:, t], state)
        outputs.append(out_t)
    return torch.stack(outputs, dim=1).to(value.dtype), None


class TestTpuQwen35Gdn(unittest.TestCase):
    def test_no_future_annotations_and_jax_op_concrete_annotations(self):
        source = (REPO_ROOT / "tpu" / "tpu_qwen35.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        future_imports = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "__future__"
            for alias in node.names
        ]
        self.assertNotIn("annotations", future_imports)

        class FakeJaxArray:
            def __init__(self, arr):
                self.arr = arr
                self.dtype = arr.dtype

            def astype(self, dtype):
                return FakeJaxArray(self.arr.to(dtype))

        backend_calls = []

        fake_jax = SimpleNamespace(
            Array=FakeJaxArray,
            default_backend=lambda: "cpu",
        )
        observed_op_calls = []
        fake_registered = []

        def fake_chunk_gdn(q, k, v, g, beta, **kwargs):
            backend_calls.append(fake_jax.default_backend())
            observed_op_calls.append((q.arr.shape, kwargs))
            return v

        def fake_jax_op(name, fn):
            sig = inspect.signature(inspect.unwrap(fn), eval_str=True)
            for param_name, param in sig.parameters.items():
                self.assertIs(
                    param.annotation,
                    FakeJaxArray,
                    f"Parameter {param_name} must annotate concrete jax.Array, not {param.annotation!r}",
                )
            self.assertIs(sig.return_annotation, FakeJaxArray)

            class FakeOp:
                def __init__(self, op_name, impl):
                    self.name = op_name
                    self.impl = impl
                    self.fake_fn = None

                def register_fake(self, fake_fn):
                    self.fake_fn = fake_fn
                    return fake_fn

                def __call__(self, q, k, v, g, beta):
                    res = self.impl(
                        FakeJaxArray(q),
                        FakeJaxArray(k),
                        FakeJaxArray(v),
                        FakeJaxArray(g),
                        FakeJaxArray(beta),
                    )
                    return res.arr

            op_obj = FakeOp(name, fn)
            fake_registered.append(op_obj)
            return op_obj

        op = tpu_qwen35.register_chunk_gdn_pallas_op(
            qk_l2norm=True,
            precision="bf16",
            jax_op_fn=fake_jax_op,
            jax_module=fake_jax,
            chunk_gdn_fn=fake_chunk_gdn,
        )
        self.assertEqual(op.name, "pallas::qwen35_chunk_gdn_l2_bf16")
        self.assertIsNotNone(op.fake_fn)
        dummy = torch.zeros((1, 128, 32, 128), dtype=torch.bfloat16)
        fake_out = op.fake_fn(dummy, dummy, dummy, dummy[..., 0].float(), dummy[..., 0].float())
        self.assertEqual(fake_out.shape, dummy.shape)
        self.assertEqual(fake_out.dtype, torch.bfloat16)

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PALLAS_INTERPRET", None)
            op(dummy, dummy, dummy, dummy[..., 0].float(), dummy[..., 0].float())
        self.assertEqual(backend_calls, ["tpu"])
        self.assertEqual(fake_jax.default_backend(), "cpu")
        self.assertEqual(observed_op_calls[0][1]["chunk_size"], 128)
        self.assertTrue(observed_op_calls[0][1]["qk_l2norm"])
        self.assertTrue(observed_op_calls[0][1]["is_inference"])

        backend_calls.clear()
        with mock.patch.dict(os.environ, {"PALLAS_INTERPRET": "1"}):
            op(dummy, dummy, dummy, dummy[..., 0].float(), dummy[..., 0].float())
        self.assertEqual(backend_calls, ["cpu"])

    def test_select_gdn_bucket_and_validation(self):
        self.assertEqual(tpu_qwen35.select_gdn_bucket(1), 128)
        self.assertEqual(tpu_qwen35.select_gdn_bucket(128), 128)
        self.assertEqual(tpu_qwen35.select_gdn_bucket(129), 256)
        self.assertEqual(tpu_qwen35.select_gdn_bucket(329), 384)
        self.assertEqual(tpu_qwen35.select_gdn_bucket(1025), 1152)
        self.assertEqual(tpu_qwen35.normalize_gdn_buckets("512, 128, 512"), (128, 512))
        with self.assertRaises(ValueError):
            tpu_qwen35.select_gdn_bucket(0)
        with self.assertRaises(ValueError):
            tpu_qwen35.normalize_gdn_buckets((100, 256))

    def test_causal_right_padding_matches_unpadded_exactly(self):
        torch.manual_seed(7)
        q = torch.randn((1, 137, 4, 32), dtype=torch.float32)
        k = torch.randn((1, 137, 4, 32), dtype=torch.float32)
        v = torch.randn((1, 137, 8, 32), dtype=torch.float32)
        g = -torch.nn.functional.softplus(torch.randn((1, 137, 8)) - 1.0)
        beta = torch.sigmoid(torch.randn((1, 137, 8)))

        ref_unpadded, _ = _naive_causal_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        padded_out, _ = tpu_qwen35.run_chunk_gdn_pallas(
            q,
            k,
            v,
            g,
            beta,
            qk_l2norm=True,
            buckets=(256,),
            precision="fp32",
            op=lambda qp, kp, vp, gp, betap: _naive_causal_delta_rule(
                qp, kp, vp, gp, betap, use_qk_l2norm_in_kernel=True
            )[0],
        )
        torch.testing.assert_close(padded_out, ref_unpadded, atol=1e-6, rtol=1e-6)

    def test_patch_module_class_instance_and_fallbacks(self):
        seen_op_shapes = []

        def fake_op(q, k, v, g, beta):
            seen_op_shapes.append((tuple(q.shape), tuple(v.shape), q.dtype, g.dtype, beta.dtype))
            return _naive_causal_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)[0]

        fake_mod = ModuleType("fake_modeling_qwen3_5")
        orig_calls = []

        def orig_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=64,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            **kwargs,
        ):
            orig_calls.append(
                {
                    "shape": tuple(query.shape),
                    "chunk_size": chunk_size,
                    "initial_state": initial_state is not None,
                    "output_final_state": bool(output_final_state),
                }
            )
            return torch.zeros_like(value), (
                torch.zeros((value.shape[0], value.shape[2], 128, 128)) if output_final_state else None
            )

        fake_mod.torch_chunk_gated_delta_rule = orig_rule

        class FakeGatedDeltaNet(torch.nn.Module):
            chunk_gated_delta_rule = staticmethod(orig_rule)

            def __init__(self, bind_instance: bool = False):
                super().__init__()
                if bind_instance:
                    self.chunk_gated_delta_rule = orig_rule

            def forward_via_attr(self, q, k, v, g, beta):
                return self.chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)

            def forward_via_global(self, q, k, v, g, beta):
                return fake_mod.torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)

        fake_mod.Qwen3_5GatedDeltaNet = FakeGatedDeltaNet

        class Container(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cls_layer = FakeGatedDeltaNet(bind_instance=False)
                self.inst_layer = FakeGatedDeltaNet(bind_instance=True)

        model = Container()
        handle = tpu_qwen35.install_qwen35_gdn_pallas_patch(
            model=model,
            modeling_module=fake_mod,
            buckets=(128, 256, 384),
            op_resolver=lambda **_kw: fake_op,
            validate_op=False,
            require_tpu_device=False,
        )

        q = torch.randn((1, 200, 16, 128), dtype=torch.bfloat16)
        k = torch.randn((1, 200, 16, 128), dtype=torch.bfloat16)
        v = torch.randn((1, 200, 32, 128), dtype=torch.bfloat16)
        g = -torch.ones((1, 200, 32), dtype=torch.float32) * 0.2
        beta = torch.full((1, 200, 32), 0.5, dtype=torch.bfloat16)

        # Call via class staticmethod, instance binding, and module global: all should route to Pallas.
        out_cls, state_cls = model.cls_layer.forward_via_attr(q, k, v, g, beta)
        out_inst, state_inst = model.inst_layer.forward_via_attr(q, k, v, g, beta)
        out_glob, state_glob = model.cls_layer.forward_via_global(q, k, v, g, beta)

        self.assertIsNone(state_cls)
        self.assertIsNone(state_inst)
        self.assertIsNone(state_glob)
        self.assertEqual(out_cls.shape, (1, 200, 32, 128))
        self.assertEqual(out_cls.dtype, torch.bfloat16)
        torch.testing.assert_close(out_cls, out_inst)
        torch.testing.assert_close(out_cls, out_glob)
        self.assertEqual(len(seen_op_shapes), 3)
        self.assertEqual(
            seen_op_shapes[0],
            ((1, 256, 32, 128), (1, 256, 32, 128), torch.bfloat16, torch.float32, torch.float32),
        )
        self.assertEqual(handle.stats["pallas_calls"], 3)
        self.assertEqual(handle.stats["fallback_calls"], 0)

        # Fallback cases must delegate safely to orig_rule.
        fake_mod.torch_chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
        fake_mod.torch_chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state=torch.zeros((1, 32, 128, 128))
        )
        fake_mod.torch_chunk_gated_delta_rule(q[..., :64], k[..., :64], v[..., :64], g, beta)
        self.assertEqual(handle.stats["fallback_calls"], 3)
        self.assertEqual(
            handle.stats["fallback_reasons"],
            {"output_final_state": 1, "initial_state": 1, "head_dim_not_128": 1},
        )
        self.assertEqual(len(orig_calls), 3)

        # Restore removes all patches cleanly.
        handle.restore()
        self.assertIs(fake_mod.torch_chunk_gated_delta_rule, orig_rule)
        self.assertIs(inspect.getattr_static(FakeGatedDeltaNet, "chunk_gated_delta_rule").__func__, orig_rule)
        model.inst_layer.forward_via_attr(q, k, v, g, beta)
        self.assertEqual(len(orig_calls), 4)

    def test_persistent_prompt_cache_and_staged_pipeline_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "prompt_cache"
            emb = torch.randn((1, 17, 64), dtype=torch.bfloat16)
            mask = torch.ones((1, 17), dtype=torch.bool)
            saved = tpu_qwen35.save_cached_prompt_embed(cache_dir, "org/model-a", "hello", 512, emb, mask)
            self.assertTrue(saved.is_file())

            hit = tpu_qwen35.load_cached_prompt_embed(cache_dir, "org/model-a", "hello", 512)
            self.assertIsNotNone(hit)
            torch.testing.assert_close(hit[0], emb[0])
            self.assertEqual(hit[1].dtype, torch.bool)

            # Model / length / prompt mismatches miss cleanly.
            self.assertIsNone(tpu_qwen35.load_cached_prompt_embed(cache_dir, "org/model-b", "hello", 512))
            self.assertIsNone(tpu_qwen35.load_cached_prompt_embed(cache_dir, "org/model-a", "hello", 256))

            # Corrupt file returns None rather than raising.
            saved.write_bytes(b"corrupt")
            self.assertIsNone(tpu_qwen35.load_cached_prompt_embed(cache_dir, "org/model-a", "hello", 512))

            # StagedTPUImagePipeline hits disk cache without calling base encode_prompt or moving text_encoder.
            encode_calls = []

            class DummyBasePipe:
                def __init__(self):
                    self.transformer = torch.nn.Linear(4, 4)
                    self.vae = torch.nn.Linear(4, 4)
                    self.text_encoder = torch.nn.Linear(4, 4)
                    self.tokenizer = object()
                    self.scheduler = object()
                    self.components = {}

                def encode_prompt(self, prompt, max_sequence_length=4096):
                    encode_calls.append(prompt)
                    return torch.ones((1, 9, 64)), torch.ones((1, 9), dtype=torch.bool)

            pipe1 = tpu_runtime.StagedTPUImagePipeline(
                DummyBasePipe(),
                __file__,
                device=torch.device("cpu"),
                prompt_cache_dir=cache_dir,
                prompt_cache_model_id="org/model-a",
            )
            out1_e, out1_m = pipe1.encode_prompt("persisted prompt", max_sequence_length=128)
            self.assertEqual(encode_calls, ["persisted prompt"])
            self.assertEqual(out1_e.shape, (1, 9, 64))

            # Second fresh pipeline instance hits disk cache with 0 base encode calls and even without text_encoder.
            base2 = DummyBasePipe()
            base2.text_encoder = None
            base2.tokenizer = None
            pipe2 = tpu_runtime.StagedTPUImagePipeline(
                base2,
                __file__,
                device=torch.device("cpu"),
                prompt_cache_dir=cache_dir,
                prompt_cache_model_id="org/model-a",
            )
            out2_e, out2_m = pipe2.encode_prompt("persisted prompt", max_sequence_length=128)
            self.assertEqual(encode_calls, ["persisted prompt"])
            torch.testing.assert_close(out2_e, out1_e)
            torch.testing.assert_close(out2_m, out1_m)

    def test_run_tpu_encode_bench_and_encode_only_prompt_cache(self):
        cli = _load_script_module("tpu/run_tpu.py", "run_tpu_qwen35_test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bundle"
            (root / "text_encoder").mkdir(parents=True)
            (root / "tokenizer").mkdir(parents=True)
            (root / "pipeline_moss_image.py").write_text("# dummy\n", encoding="utf-8")

            moved_devices = []
            encode_history = []
            load_calls = []

            class FakeEncoder(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.w = torch.nn.Parameter(torch.zeros(2, 2))

                def to(self, device, *args, **kwargs):
                    moved_devices.append(str(device))
                    return super().to(device, *args, **kwargs)

            class FakePipe:
                def __init__(self, *, text_encoder, tokenizer, **_kwargs):
                    self.text_encoder = text_encoder
                    self.tokenizer = tokenizer

                def encode_prompt(self, prompt, max_sequence_length=4096):
                    encode_history.append((prompt, max_sequence_length))
                    seq_len = 11 if prompt else 3
                    return torch.ones((1, seq_len, 8)), torch.ones((1, seq_len), dtype=torch.bool)

            fake_module = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_a, **_kw: object()),
                AutoModel=SimpleNamespace(
                    from_pretrained=lambda *_a, **_kw: (load_calls.append(1) or FakeEncoder())
                ),
                MOSSImageScheduler=lambda: object(),
                MOSSImagePipeline=FakePipe,
            )

            report_path = Path(tmp) / "encode_bench.json"
            embeds_path = Path(tmp) / "embeds.pt"
            cache_dir = Path(tmp) / "cache"

            fake_handle = SimpleNamespace(
                stats={"pallas_calls": 24, "fallback_calls": 0, "fallback_reasons": {}},
                restore=lambda: None,
            )

            with (
                mock.patch.object(tpu_runtime, "is_available", return_value=False),
                mock.patch.object(tpu_runtime, "resolve_model_bundle", return_value=root),
                mock.patch.object(tpu_runtime, "_import_pipeline_module", return_value=(fake_module, root)),
                mock.patch.object(
                    tpu_qwen35,
                    "qwen35_gdn_pallas_context",
                    side_effect=lambda enabled=True, **_kw: nullcontext(fake_handle if enabled else None),
                ),
            ):
                rc = cli.main(
                    [
                        "--mode",
                        "encode-bench",
                        "--bench-modes",
                        "eager,defer_and_fuse+gdn_pallas",
                        "--runs",
                        "2",
                        "--prompt",
                        "A ceramic teapot",
                        "--warm-prompt",
                        "A glass vase",
                        "--save-report",
                        str(report_path),
                        "--allow-cpu",
                    ]
                )
                self.assertEqual(rc, 0)
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["mode"], "encode-bench")
                self.assertIn("tokenizer_load_seconds", report)
                self.assertIn("model_load_seconds", report)
                eager_stats = report["results"]["eager"]
                self.assertEqual(eager_stats["prompt_tokens"], 11)
                self.assertEqual(eager_stats["negative_tokens"], 3)
                self.assertEqual(len(eager_stats["warm_encode_seconds"]), 2)
                self.assertTrue(eager_stats["warm_prompt_distinct"])
                self.assertIsNone(eager_stats["gdn_pallas_stats"])

                pallas_stats = report["results"]["defer_and_fuse+gdn_pallas"]
                self.assertEqual(pallas_stats["eager_mode"], "defer_and_fuse")
                self.assertTrue(pallas_stats["gdn_pallas"])
                self.assertEqual(pallas_stats["gdn_pallas_stats"]["pallas_calls"], 24)
                self.assertIn("cpu", moved_devices)

                # Test encode-only with persistent prompt cache: 1st run loads, 2nd run skips loading weights.
                load_calls.clear()
                encode_history.clear()
                rc1 = cli.main(
                    [
                        "--mode",
                        "encode-only",
                        "--prompt",
                        "Cached prompt",
                        "--prompt-cache-dir",
                        str(cache_dir),
                        "--save-embeds",
                        str(embeds_path),
                        "--allow-cpu",
                    ]
                )
                self.assertEqual(rc1, 0)
                self.assertEqual(len(load_calls), 1)
                self.assertEqual(len(encode_history), 2)

                rc2 = cli.main(
                    [
                        "--mode",
                        "encode-only",
                        "--prompt",
                        "Cached prompt",
                        "--prompt-cache-dir",
                        str(cache_dir),
                        "--save-embeds",
                        str(embeds_path),
                        "--allow-cpu",
                    ]
                )
                self.assertEqual(rc2, 0)
                self.assertEqual(len(load_calls), 1)
                self.assertEqual(len(encode_history), 2)

    def test_bench_qwen35_gdn_script_dry_run_and_metrics(self):
        bench = _load_script_module("tpu/bench_qwen35_gdn_tpu.py", "bench_qwen35_gdn_tpu_test")

        ref = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        cand = torch.tensor([[[[1.01, 1.99], [3.0, 4.02]]]])
        metrics = bench.compute_parity_metrics(ref, cand)
        self.assertAlmostEqual(metrics["max_abs_diff"], 0.02, places=5)
        self.assertGreater(metrics["cosine_similarity"], 0.999)
        self.assertFalse(metrics["has_nan_or_inf"])

        fake_mod = ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
        fake_mod.torch_chunk_gated_delta_rule = (
            lambda q, k, v, g, beta, **_kw: _naive_causal_delta_rule(
                q, k, v, g, beta, use_qk_l2norm_in_kernel=True
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "gdn_dry.json"
            orig_import = importlib.import_module

            def _fake_import(name, *args, **kwargs):
                if name == "transformers.models.qwen3_5.modeling_qwen3_5":
                    return fake_mod
                return orig_import(name, *args, **kwargs)

            with (
                mock.patch.object(tpu_runtime, "is_available", return_value=False),
                mock.patch.object(importlib, "import_module", side_effect=_fake_import),
            ):
                rc = bench.main(
                    [
                        "--allow-cpu",
                        "--skip-pallas",
                        "--seq-len",
                        "16",
                        "--num-k-heads",
                        "2",
                        "--num-v-heads",
                        "4",
                        "--head-dim",
                        "16",
                        "--warmup",
                        "1",
                        "--iters",
                        "2",
                        "--save-report",
                        str(report_path),
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("hf_eager", payload["recurrence_ab"])
            self.assertIn("hf_defer_and_fuse", payload["recurrence_ab"])
            self.assertEqual(
                payload["recurrence_ab"]["hf_eager"]["parity_vs_cpu_fp32"]["has_nan_or_inf"],
                False,
            )

    def test_staged_pipeline_split_text_and_diffusion_eager_modes(self):
        class FakeBasePipe:
            def __init__(self):
                self.transformer = torch.nn.Linear(4, 4)
                self.vae = torch.nn.Linear(4, 4)
                self.text_encoder = torch.nn.Linear(4, 4)
                self.tokenizer = object()
                self.scheduler = object()
                self.components = {}
                self.forward_calls = []

            def encode_prompt(self, prompt, max_sequence_length=4096):
                return torch.ones((1, 4, 8)), torch.ones((1, 4), dtype=torch.bool)

            def __call__(self, **kwargs):
                self.forward_calls.append(kwargs)
                return SimpleNamespace(images=["ok"])

        seen_modes = []
        from contextlib import nullcontext

        def _record_eager(mode, device):
            seen_modes.append(mode)
            return nullcontext()

        with mock.patch.object(tpu_runtime, "configure_eager_mode", side_effect=_record_eager):
            inherited = tpu_runtime.StagedTPUImagePipeline(
                FakeBasePipe(),
                __file__,
                device=torch.device("cpu"),
                offload="none",
                eager_mode="defer_and_fuse",
                prompt_buckets=(8,),
            )
            self.assertEqual(inherited.eager_mode, "defer_and_fuse")
            self.assertEqual(inherited.text_eager_mode, "defer_and_fuse")

            with self.assertRaises(ValueError):
                tpu_runtime.StagedTPUImagePipeline(
                    FakeBasePipe(),
                    __file__,
                    device=torch.device("cpu"),
                    eager_mode="defer_and_fuse",
                    text_eager_mode="invalid",
                )

            pipe = tpu_runtime.StagedTPUImagePipeline(
                FakeBasePipe(),
                __file__,
                device=torch.device("cpu"),
                offload="none",
                eager_mode="defer_and_fuse",
                text_eager_mode="default",
                prompt_buckets=(8,),
            )
            self.assertEqual(pipe.eager_mode, "defer_and_fuse")
            self.assertEqual(pipe.text_eager_mode, "default")

            seen_modes.clear()
            pipe.encode_prompt("alpha")
            self.assertEqual(seen_modes, ["default"])

            seen_modes.clear()
            pipe("beta", negative_prompt="neg", num_inference_steps=2, guidance_scale=2.0)
            self.assertEqual(seen_modes, ["default", "defer_and_fuse"])

            seen_modes.clear()
            pre = torch.ones((1, 4, 8))
            mask = torch.ones((1, 4), dtype=torch.bool)
            pipe(
                prompt_embeds=pre,
                prompt_attention_mask=mask,
                negative_prompt_embeds=pre,
                negative_prompt_attention_mask=mask,
                num_inference_steps=2,
                guidance_scale=2.0,
            )
            self.assertEqual(seen_modes, ["defer_and_fuse"])

    def test_run_tpu_cli_text_eager_mode_resolution(self):
        cli = _load_script_module("tpu/run_tpu.py", "run_tpu_text_eager_test")
        defaults = cli.build_parser().parse_args([])
        self.assertEqual(defaults.eager_mode, "default")
        self.assertEqual(defaults.text_eager_mode, "auto")

        self.assertEqual(
            cli._resolve_text_eager_mode("auto", eager_mode="defer_and_fuse", gdn_pallas=True),
            "default",
        )
        self.assertEqual(
            cli._resolve_text_eager_mode("auto", eager_mode="defer_and_fuse", gdn_pallas=False),
            "defer_and_fuse",
        )
        self.assertEqual(
            cli._resolve_text_eager_mode("default", eager_mode="defer_and_fuse", gdn_pallas=False),
            "default",
        )
        self.assertEqual(
            cli._resolve_text_eager_mode("defer_and_fuse", eager_mode="defer_and_fuse", gdn_pallas=True),
            "defer_and_fuse",
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.png"
            report_path = Path(tmp) / "report.json"
            load_kwargs = []

            class FakeLoadedPipe:
                gdn_pallas_stats = None

                def progress_bar(self, iterable):
                    return iterable

                def __call__(self, **_kwargs):
                    from PIL import Image

                    list(self.progress_bar([1, 2]))
                    return SimpleNamespace(images=[Image.new("RGB", (16, 16))])

                def disable_gdn_pallas(self):
                    pass

            def _fake_load(*_args, **kwargs):
                load_kwargs.append(kwargs)
                return FakeLoadedPipe()

            with (
                mock.patch.object(tpu_runtime, "is_available", return_value=False),
                mock.patch.object(tpu_runtime, "load_pipeline", side_effect=_fake_load),
            ):
                rc = cli.main(
                    [
                        "--mode",
                        "generate",
                        "--eager-mode",
                        "defer_and_fuse",
                        "--gdn-pallas",
                        "--allow-cpu",
                        "--output",
                        str(out_path),
                        "--save-report",
                        str(report_path),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(load_kwargs[0]["eager_mode"], "defer_and_fuse")
            self.assertEqual(load_kwargs[0]["text_eager_mode"], "default")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["eager_mode"], "defer_and_fuse")
            self.assertEqual(report["text_eager_mode"], "default")
            self.assertTrue(report["text_encoder_loaded"])
            self.assertTrue(report["gdn_pallas_requested"])
            self.assertTrue(report["gdn_pallas"])
            self.assertFalse(report["gdn_pallas_active"])

    def test_generate_with_cached_prompts_reports_gdn_pallas_inactive(self):
        cli = _load_script_module("tpu/run_tpu.py", "run_tpu_cached_report_test")
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "prompt_cache"
            out_path = Path(tmp) / "out.png"
            report_path = Path(tmp) / "report.json"
            model_id = "OpenMOSS-Team/MOSS-Image-Base"
            pos_embeds = torch.zeros((1, 4, 8), dtype=torch.bfloat16)
            pos_mask = torch.ones((1, 4), dtype=torch.bool)
            neg_embeds = torch.ones((1, 3, 8), dtype=torch.bfloat16)
            neg_mask = torch.ones((1, 3), dtype=torch.bool)
            tpu_qwen35.save_cached_prompt_embed(
                cache_dir, model_id, "cached prompt", 512, pos_embeds, pos_mask
            )
            tpu_qwen35.save_cached_prompt_embed(
                cache_dir, model_id, "", 512, neg_embeds, neg_mask
            )

            load_kwargs = []

            class FakeLoadedPipe:
                gdn_pallas_stats = None

                def progress_bar(self, iterable):
                    return iterable

                def __call__(self, **_kwargs):
                    from PIL import Image

                    list(self.progress_bar([1]))
                    return SimpleNamespace(images=[Image.new("RGB", (16, 16))])

                def disable_gdn_pallas(self):
                    pass

            def _fake_load(*_args, **kwargs):
                load_kwargs.append(kwargs)
                return FakeLoadedPipe()

            with (
                mock.patch.object(tpu_runtime, "is_available", return_value=False),
                mock.patch.object(tpu_runtime, "load_pipeline", side_effect=_fake_load),
                mock.patch("builtins.print"),
            ):
                rc = cli.main(
                    [
                        "--mode",
                        "generate",
                        "--model",
                        model_id,
                        "--prompt",
                        "cached prompt",
                        "--negative-prompt",
                        "",
                        "--max-sequence-length",
                        "512",
                        "--prompt-cache-dir",
                        str(cache_dir),
                        "--gdn-pallas",
                        "--allow-cpu",
                        "--output",
                        str(out_path),
                        "--save-report",
                        str(report_path),
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertFalse(load_kwargs[0]["include_text_encoder"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["text_encoder_loaded"])
            self.assertEqual(report["prompt_cache_hits"], {"positive": True, "negative": True})
            self.assertTrue(report["gdn_pallas_requested"])
            self.assertFalse(report["gdn_pallas"])
            self.assertFalse(report["gdn_pallas_active"])
            self.assertIsNone(report["gdn_pallas_stats"])

    def test_pallas_missing_prereq_error_explains_python312_jax_and_fallback(self):
        with mock.patch.dict(tpu_qwen35._GDN_OP_CACHE, {}, clear=True):
            real_import = tpu_qwen35.importlib.import_module

            def _missing_jax(name, *args, **kwargs):
                if name == "jax":
                    raise ImportError("No module named 'jax'")
                return real_import(name, *args, **kwargs)

            with mock.patch.object(tpu_qwen35.importlib, "import_module", side_effect=_missing_jax):
                with self.assertRaises(RuntimeError) as ctx_jax:
                    tpu_qwen35.register_chunk_gdn_pallas_op()
            msg_jax = str(ctx_jax.exception)
            self.assertIn("JAX", msg_jax)
            self.assertIn("Python >= 3.12", msg_jax)
            self.assertIn("omit `--gdn-pallas`", msg_jax)

            def _missing_deltanet(name, *args, **kwargs):
                if name == "deltanet_tpu":
                    raise ImportError("No module named 'deltanet_tpu'")
                return real_import(name, *args, **kwargs)

            with mock.patch.object(
                tpu_qwen35.importlib, "import_module", side_effect=_missing_deltanet
            ):
                with self.assertRaises(RuntimeError) as ctx_dn:
                    tpu_qwen35.register_chunk_gdn_pallas_op(jax_module=SimpleNamespace(Array=object))
            msg_dn = str(ctx_dn.exception)
            self.assertIn("deltanet_tpu", msg_dn)
            self.assertIn("Python >= 3.12", msg_dn)
            self.assertIn("omit `--gdn-pallas`", msg_dn)


if __name__ == "__main__":
    unittest.main()
