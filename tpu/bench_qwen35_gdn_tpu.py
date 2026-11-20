#!/usr/bin/env python3
"""A/B benchmark for Qwen3.5 Gated DeltaNet: HF PyTorch recurrence vs deltanet-tpu Pallas.

Note for Colab TorchTPU environments:
  Install deltanet-tpu without upgrading JAX or libtpu:
    pip install --no-deps deltanet-tpu
  Do NOT run `pip install deltanet-tpu[tpu]`, which can overwrite TorchTPU's pinned libtpu.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tpu import tpu_qwen35, tpu_runtime  # noqa: E402


def compute_parity_metrics(ref: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Compute FP32 parity statistics between reference and candidate tensors."""
    ref_f32 = ref.detach().cpu().to(torch.float32)
    cand_f32 = candidate.detach().cpu().to(torch.float32)
    diff = (ref_f32 - cand_f32).abs()
    ref_norm = float(torch.linalg.vector_norm(ref_f32).item())
    diff_norm = float(torch.linalg.vector_norm(ref_f32 - cand_f32).item())
    cos = float(
        F.cosine_similarity(ref_f32.reshape(1, -1), cand_f32.reshape(1, -1), dim=-1).item()
    )
    return {
        "max_abs_diff": round(float(diff.max().item()), 6),
        "mean_abs_diff": round(float(diff.mean().item()), 6),
        "rel_l2_error": round(diff_norm / max(ref_norm, 1e-12), 6),
        "cosine_similarity": round(cos, 6),
        "has_nan_or_inf": bool(not torch.isfinite(cand_f32).all().item()),
    }


def _benchmark_callable(fn, *, warmup: int, iters: int) -> tuple[torch.Tensor, float, list[float]]:
    tpu_runtime.synchronize()
    t0 = time.perf_counter()
    first_out = fn()
    tpu_runtime.synchronize()
    cold_seconds = time.perf_counter() - t0

    for _ in range(warmup):
        _ = fn()
    tpu_runtime.synchronize()

    latencies_ms = []
    out = first_out
    for _ in range(iters):
        t_iter0 = time.perf_counter()
        out = fn()
        tpu_runtime.synchronize()
        latencies_ms.append(round((time.perf_counter() - t_iter0) * 1000.0, 3))

    return out.detach().cpu(), cold_seconds, latencies_ms


def _summarize_latencies(cold_seconds: float, latencies_ms: list[float], ref: torch.Tensor, out: torch.Tensor) -> dict:
    return {
        "cold_compile_seconds": round(cold_seconds, 4),
        "warm_median_ms": round(statistics.median(latencies_ms), 3),
        "warm_mean_ms": round(statistics.mean(latencies_ms), 3),
        "warm_min_ms": round(min(latencies_ms), 3),
        "warm_latencies_ms": latencies_ms,
        "parity_vs_cpu_fp32": compute_parity_metrics(ref, out),
    }


def _build_synthetic_gdn_inputs(
    *,
    batch_size: int,
    seq_len: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((batch_size, seq_len, num_k_heads, head_dim), generator=gen, dtype=torch.bfloat16)
    k = torch.randn((batch_size, seq_len, num_k_heads, head_dim), generator=gen, dtype=torch.bfloat16)
    v = torch.randn((batch_size, seq_len, num_v_heads, head_dim), generator=gen, dtype=torch.bfloat16)
    # Match Qwen3_5GatedDeltaNet: negative log decay g and sigmoid gate beta.
    a_raw = torch.randn((batch_size, seq_len, num_v_heads), generator=gen, dtype=torch.float32)
    g = -F.softplus(a_raw - 1.0)
    beta = torch.sigmoid(torch.randn((batch_size, seq_len, num_v_heads), generator=gen, dtype=torch.float32)).to(
        torch.bfloat16
    )
    if num_k_heads != num_v_heads:
        repeat_factor = num_v_heads // num_k_heads
        q = q.repeat_interleave(repeat_factor, dim=2)
        k = k.repeat_interleave(repeat_factor, dim=2)
    return q, k, v, g, beta


def _resolve_device(allow_cpu: bool) -> torch.device:
    if tpu_runtime.is_available():
        return tpu_runtime.tpu_device()
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("No TPU device detected. Run on TorchTPU Colab or pass --allow-cpu for dry checks.")


def run_benchmark(args) -> dict:
    dev = _resolve_device(args.allow_cpu)
    tpu_runtime.configure_precision(args.matmul_precision)

    modeling_qwen3_5 = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    hf_chunk_gdn = modeling_qwen3_5.torch_chunk_gated_delta_rule

    q_cpu, k_cpu, v_cpu, g_cpu, beta_cpu = _build_synthetic_gdn_inputs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_k_heads=args.num_k_heads,
        num_v_heads=args.num_v_heads,
        head_dim=args.head_dim,
        seed=args.seed,
    )

    with torch.no_grad():
        ref_out, _ = hf_chunk_gdn(
            q_cpu.float(),
            k_cpu.float(),
            v_cpu.float(),
            g_cpu.float(),
            beta_cpu.float(),
            chunk_size=64,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

    q_dev = q_cpu.to(dev)
    k_dev = k_cpu.to(dev)
    v_dev = v_cpu.to(dev)
    g_dev = g_cpu.to(dev)
    beta_dev = beta_cpu.to(dev)

    recurrence_results: dict[str, dict] = {}
    for eager_label, eager_mode in (("hf_eager", "default"), ("hf_defer_and_fuse", "defer_and_fuse")):
        with torch.no_grad(), tpu_runtime.configure_eager_mode(eager_mode, dev):

            def _run_hf():
                out, _ = hf_chunk_gdn(
                    q_dev,
                    k_dev,
                    v_dev,
                    g_dev,
                    beta_dev,
                    chunk_size=64,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                )
                return out

            out_cpu, cold_s, lats = _benchmark_callable(_run_hf, warmup=args.warmup, iters=args.iters)
            recurrence_results[eager_label] = _summarize_latencies(cold_s, lats, ref_out, out_cpu)

    buckets = (
        (tpu_qwen35.select_gdn_bucket(args.seq_len),)
        if args.bucket_len == 0
        else tpu_qwen35.normalize_gdn_buckets((args.bucket_len,))
        if args.bucket_len > 0
        else None
    )
    if not args.skip_pallas:
        with torch.no_grad(), tpu_runtime.configure_eager_mode("default", dev):

            def _run_pallas():
                out, _ = tpu_qwen35.run_chunk_gdn_pallas(
                    q_dev,
                    k_dev,
                    v_dev,
                    g_dev,
                    beta_dev,
                    qk_l2norm=True,
                    buckets=buckets,
                    precision="bf16",
                )
                return out

            pallas_out_cpu, cold_s, lats = _benchmark_callable(
                _run_pallas, warmup=args.warmup, iters=args.iters
            )
            pallas_stats = _summarize_latencies(cold_s, lats, ref_out, pallas_out_cpu)
            hf_eager_ms = recurrence_results["hf_eager"]["warm_median_ms"]
            hf_df_ms = recurrence_results["hf_defer_and_fuse"]["warm_median_ms"]
            pallas_ms = pallas_stats["warm_median_ms"]
            pallas_stats["speedup_vs_hf_eager"] = round(hf_eager_ms / max(pallas_ms, 1e-6), 2)
            pallas_stats["speedup_vs_hf_defer_and_fuse"] = round(hf_df_ms / max(pallas_ms, 1e-6), 2)
            recurrence_results["deltanet_tpu_pallas"] = pallas_stats

    layer_results = None
    if args.include_layer:
        config_cls = getattr(modeling_qwen3_5, "Qwen3_5TextConfig", None)
        layer_cls = getattr(modeling_qwen3_5, "Qwen3_5GatedDeltaNet", None)
        if config_cls is None or layer_cls is None:
            raise RuntimeError("Qwen3_5TextConfig / Qwen3_5GatedDeltaNet not found in installed transformers.")
        cfg = config_cls(
            hidden_size=4096,
            linear_num_key_heads=args.num_k_heads,
            linear_num_value_heads=args.num_v_heads,
            linear_key_head_dim=args.head_dim,
            linear_value_head_dim=args.head_dim,
            linear_conv_kernel_dim=4,
            hidden_act="silu",
            rms_norm_eps=1e-6,
        )
        layer = layer_cls(cfg, layer_idx=0).to(device=dev, dtype=torch.bfloat16).eval()
        hidden_cpu = torch.randn(
            (args.batch_size, args.seq_len, cfg.hidden_size),
            generator=torch.Generator(device="cpu").manual_seed(args.seed + 1),
            dtype=torch.bfloat16,
        )
        hidden_dev = hidden_cpu.to(dev)
        with torch.no_grad():
            layer_results = {}
            for label, eager_mode in (("layer_hf_eager", "default"), ("layer_hf_defer_and_fuse", "defer_and_fuse")):
                with tpu_runtime.configure_eager_mode(eager_mode, dev):
                    out_l, cold_s, lats = _benchmark_callable(
                        lambda: layer(hidden_dev), warmup=args.warmup, iters=args.iters
                    )
                    layer_results[label] = {
                        "cold_compile_seconds": round(cold_s, 4),
                        "warm_median_ms": round(statistics.median(lats), 3),
                        "warm_mean_ms": round(statistics.mean(lats), 3),
                        "_ref_tensor": out_l,
                    }

            base_layer_ref = layer_results["layer_hf_defer_and_fuse"].pop("_ref_tensor")
            layer_results["layer_hf_eager"].pop("_ref_tensor")

            if not args.skip_pallas:
                with (
                    tpu_runtime.configure_eager_mode("default", dev),
                    tpu_qwen35.qwen35_gdn_pallas_context(
                        enabled=True,
                        model=layer,
                        buckets=buckets,
                        require_tpu_device=(dev.type == "tpu"),
                    ) as patch_handle,
                ):
                    pallas_layer_out, cold_s, lats = _benchmark_callable(
                        lambda: layer(hidden_dev), warmup=args.warmup, iters=args.iters
                    )
                    layer_results["layer_deltanet_tpu_pallas"] = {
                        "cold_compile_seconds": round(cold_s, 4),
                        "warm_median_ms": round(statistics.median(lats), 3),
                        "warm_mean_ms": round(statistics.mean(lats), 3),
                        "gdn_pallas_stats": dict(patch_handle.stats),
                        "parity_vs_layer_hf_defer_and_fuse": compute_parity_metrics(
                            base_layer_ref, pallas_layer_out
                        ),
                    }

    report = {
        "device": tpu_runtime.device_name() or str(dev),
        "shapes": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "bucket_len": buckets[0] if buckets else None,
            "num_k_heads": args.num_k_heads,
            "num_v_heads": args.num_v_heads,
            "head_dim": args.head_dim,
        },
        "recurrence_ab": recurrence_results,
        "layer_ab": layer_results,
    }
    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=329)
    parser.add_argument(
        "--bucket-len",
        type=int,
        default=0,
        help="Static sequence bucket length (multiple of 128). 0=auto via DEFAULT_GDN_BUCKETS, -1=disable.",
    )
    parser.add_argument("--num-k-heads", type=int, default=16)
    parser.add_argument("--num-v-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matmul-precision", choices=tpu_runtime.VALID_MATMUL_PRECISIONS, default="high")
    parser.add_argument("--include-layer", action="store_true", help="Also benchmark 1 Qwen3_5GatedDeltaNet block.")
    parser.add_argument("--skip-pallas", action="store_true", help="Run only the HF PyTorch reference/baseline.")
    parser.add_argument("--save-report", default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
