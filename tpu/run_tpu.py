#!/usr/bin/env python3
"""Run or benchmark MOSS-Image on Google Cloud TPU via TorchTPU (torch_tpu)."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_runtime import DEFAULT_MODEL  # noqa: E402
from tpu import tpu_qwen35, tpu_runtime  # noqa: E402


def _parse_buckets(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"Invalid bucket list: {raw!r}")
    return values


def _require_tpu_or_override(allow_cpu: bool) -> torch.device:
    if tpu_runtime.is_available():
        return tpu_runtime.tpu_device()
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError(
        "Google Cloud TPU / torch_tpu accelerator was not detected. "
        "Run on a Colab TPU runtime (with torch_tpu installed) or pass --allow-cpu for dry checks."
    )


def run_probe(args) -> dict:
    info = {
        "torch_version": torch.__version__,
        "accelerator_tpu_available": tpu_runtime.is_available(),
        "device_name": tpu_runtime.device_name(),
        "tpu_budget_gib": round(tpu_runtime.tpu_budget_gib(), 2),
        "pallas_available": hasattr(getattr(torch, "tpu", None), "pallas"),
        "eager_modes": (
            list(tpu_runtime.VALID_EAGER_MODES)
            if hasattr(getattr(torch, "tpu", None), "set_eager_mode")
            else []
        ),
    }
    if tpu_runtime.is_available() or args.allow_cpu:
        dev = _require_tpu_or_override(args.allow_cpu)
        tpu_runtime.configure_precision(args.matmul_precision)
        with tpu_runtime.configure_eager_mode(args.eager_mode, dev):
            t0 = time.perf_counter()
            a = torch.randn((512, 512), device=dev, dtype=torch.bfloat16)
            b = torch.randn((512, 512), device=dev, dtype=torch.bfloat16)
            c = a @ b
            _ = float(c[0, 0].float().item())
            tpu_runtime.synchronize()
        info["eager_mode"] = args.eager_mode if dev.type == "tpu" else None
        info["matmul_smoke_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        info["peak_hbm_gib"] = tpu_runtime.max_memory_allocated_gib()
    print(json.dumps(info, indent=2))
    return info


def _resolve_gdn_buckets(args) -> tuple[int, ...] | None:
    raw = getattr(args, "gdn_buckets", None)
    if raw is None:
        return tpu_qwen35.DEFAULT_GDN_BUCKETS
    return tpu_qwen35.normalize_gdn_buckets(raw)


def _load_text_encoder_components(args, dev: torch.device):
    root = tpu_runtime.resolve_model_bundle(
        args.model, local_files_only=args.local_files_only, components=tpu_runtime.TEXT_COMPONENTS
    )
    module, _ = tpu_runtime._import_pipeline_module(root)

    root_path = Path(root)
    tokenizer_path = root_path / "tokenizer"
    if not tokenizer_path.is_dir():
        tokenizer_path = root_path / "text_encoder"

    common = {"local_files_only": args.local_files_only}
    t_tok0 = time.perf_counter()
    tokenizer = module.AutoTokenizer.from_pretrained(tokenizer_path, **common)
    tokenizer_load_seconds = time.perf_counter() - t_tok0

    t_mod0 = time.perf_counter()
    text_encoder = module.AutoModel.from_pretrained(
        root_path / "text_encoder", dtype=torch.bfloat16, **common
    ).eval()
    model_load_seconds = time.perf_counter() - t_mod0

    pipe = module.MOSSImagePipeline(
        transformer=None,
        scheduler=module.MOSSImageScheduler(),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=None,
    )
    pipe._tpu_execution_device = dev
    return pipe, tokenizer_load_seconds, model_load_seconds


def _timed_encode_single(pipe, text: str, max_sequence_length: int) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    tpu_runtime.synchronize()
    t_fwd0 = time.perf_counter()
    embeds, mask = pipe.encode_prompt(text, max_sequence_length=max_sequence_length)
    tpu_runtime.synchronize()
    fwd_seconds = time.perf_counter() - t_fwd0

    t_d2h0 = time.perf_counter()
    embeds_cpu = embeds.detach().cpu()
    mask_cpu = mask.detach().cpu().to(torch.bool)
    tpu_runtime.synchronize()
    d2h_seconds = time.perf_counter() - t_d2h0
    return embeds_cpu, mask_cpu, fwd_seconds, d2h_seconds


def run_encode_only(args) -> dict:
    dev = _require_tpu_or_override(args.allow_cpu)
    cache_dir = getattr(args, "prompt_cache_dir", None)
    use_gdn_pallas = bool(getattr(args, "gdn_pallas", False))
    gdn_buckets = _resolve_gdn_buckets(args)

    pos_embeds, pos_mask = (None, None)
    neg_embeds, neg_mask = (None, None)
    need_neg = args.guidance_scale != 1.0
    cache_hits = {"positive": False, "negative": False}

    if cache_dir is not None:
        cached_pos = tpu_qwen35.load_cached_prompt_embed(
            cache_dir, args.model, args.prompt, args.max_sequence_length
        )
        if cached_pos is not None:
            pos_embeds, pos_mask = cached_pos[0].unsqueeze(0), cached_pos[1].unsqueeze(0)
            cache_hits["positive"] = True
        if need_neg:
            cached_neg = tpu_qwen35.load_cached_prompt_embed(
                cache_dir, args.model, args.negative_prompt, args.max_sequence_length
            )
            if cached_neg is not None:
                neg_embeds, neg_mask = cached_neg[0].unsqueeze(0), cached_neg[1].unsqueeze(0)
                cache_hits["negative"] = True

    t0 = time.perf_counter()
    tokenizer_load_seconds = 0.0
    model_load_seconds = 0.0
    h2d_seconds = 0.0
    pos_fwd_seconds = 0.0
    neg_fwd_seconds = 0.0
    d2h_sync_seconds = 0.0
    gdn_pallas_stats = None
    peak_gib = None

    need_encode_pos = pos_embeds is None
    need_encode_neg = need_neg and neg_embeds is None
    if need_encode_pos or need_encode_neg:
        pipe, tokenizer_load_seconds, model_load_seconds = _load_text_encoder_components(args, dev)
        tpu_runtime.configure_precision(args.matmul_precision)
        text_eager_mode = _resolve_text_eager_mode(
            getattr(args, "text_eager_mode", "auto"),
            eager_mode=args.eager_mode,
            gdn_pallas=use_gdn_pallas,
        )
        with (
            tpu_runtime.configure_eager_mode(text_eager_mode, dev),
            tpu_qwen35.qwen35_gdn_pallas_context(
                enabled=use_gdn_pallas,
                model=pipe.text_encoder,
                buckets=gdn_buckets,
                require_tpu_device=(dev.type == "tpu"),
            ) as gdn_handle,
        ):
            t_h2d0 = time.perf_counter()
            pipe.text_encoder.to(dev)
            tpu_runtime.synchronize()
            h2d_seconds = time.perf_counter() - t_h2d0

            if need_encode_pos:
                pos_embeds, pos_mask, pos_fwd_seconds, pos_d2h = _timed_encode_single(
                    pipe, args.prompt, args.max_sequence_length
                )
                d2h_sync_seconds += pos_d2h
                if cache_dir is not None:
                    tpu_qwen35.save_cached_prompt_embed(
                        cache_dir, args.model, args.prompt, args.max_sequence_length, pos_embeds, pos_mask
                    )

            if need_encode_neg:
                neg_embeds, neg_mask, neg_fwd_seconds, neg_d2h = _timed_encode_single(
                    pipe, args.negative_prompt, args.max_sequence_length
                )
                d2h_sync_seconds += neg_d2h
                if cache_dir is not None:
                    tpu_qwen35.save_cached_prompt_embed(
                        cache_dir,
                        args.model,
                        args.negative_prompt,
                        args.max_sequence_length,
                        neg_embeds,
                        neg_mask,
                    )

            peak_gib = tpu_runtime.max_memory_allocated_gib()
            if gdn_handle is not None:
                gdn_pallas_stats = dict(gdn_handle.stats)

    elapsed = time.perf_counter() - t0

    out_path = Path(args.save_embeds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "prompt_embeds": pos_embeds.detach().cpu(),
        "prompt_attention_mask": pos_mask.detach().cpu().to(torch.bool),
        "negative_prompt_embeds": neg_embeds.detach().cpu() if neg_embeds is not None else None,
        "negative_prompt_attention_mask": (
            neg_mask.detach().cpu().to(torch.bool) if neg_mask is not None else None
        ),
    }
    torch.save(payload, out_path)

    summary = {
        "saved_embeds": str(out_path.resolve()),
        "prompt_tokens": int(payload["prompt_attention_mask"].sum().item()),
        "negative_tokens": (
            int(payload["negative_prompt_attention_mask"].sum().item())
            if payload["negative_prompt_attention_mask"] is not None
            else 0
        ),
        "encode_seconds": round(elapsed, 2),
        "peak_hbm_gib": peak_gib,
        "text_encoder_loaded": need_encode_pos or need_encode_neg,
        "gdn_pallas_requested": use_gdn_pallas,
        "gdn_pallas": use_gdn_pallas and (need_encode_pos or need_encode_neg),
        "gdn_pallas_active": gdn_pallas_stats is not None,
        "gdn_pallas_stats": gdn_pallas_stats,
        "prompt_cache_hits": cache_hits,
        "breakdown": {
            "tokenizer_load_seconds": round(tokenizer_load_seconds, 3),
            "model_load_seconds": round(model_load_seconds, 3),
            "h2d_transfer_seconds": round(h2d_seconds, 3),
            "pos_encode_seconds": round(pos_fwd_seconds, 3),
            "neg_encode_seconds": round(neg_fwd_seconds, 3),
            "d2h_sync_seconds": round(d2h_sync_seconds, 3),
        },
    }
    print(json.dumps(summary, indent=2))
    return summary


def _parse_encode_bench_modes(raw: str, global_gdn_pallas: bool) -> list[tuple[str, str, bool, bool]]:
    tokens = [part.strip() for part in raw.split(",") if part.strip()]
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("--bench-modes must contain one or more distinct modes.")
    spec_map = {
        "eager": ("default", False, False),
        "defer_and_fuse": ("defer_and_fuse", False, False),
        "gdn_pallas": ("default", True, False),
        "eager+gdn_pallas": ("default", True, False),
        "defer_and_fuse+gdn_pallas": ("defer_and_fuse", True, False),
        "compile": ("default", False, True),
        "compile+gdn_pallas": ("default", True, True),
    }
    parsed = []
    has_explicit_pallas = False
    for token in tokens:
        if token not in spec_map:
            raise ValueError(
                f"Unknown encode-bench mode {token!r}. Expected one of {tuple(spec_map)}."
            )
        eager_mode, use_pallas, use_compile = spec_map[token]
        has_explicit_pallas = has_explicit_pallas or use_pallas
        parsed.append((token, eager_mode, use_pallas, use_compile))
    if global_gdn_pallas and not has_explicit_pallas:
        parsed = [(name, eager_mode, True, use_compile) for name, eager_mode, _, use_compile in parsed]
    return parsed


def run_encode_bench(args) -> dict:
    dev = _require_tpu_or_override(args.allow_cpu)
    gdn_buckets = _resolve_gdn_buckets(args)
    use_gdn_global = bool(getattr(args, "gdn_pallas", False))
    mode_specs = _parse_encode_bench_modes(args.bench_modes, use_gdn_global)
    warm_prompt = getattr(args, "warm_prompt", None) or args.prompt

    pipe, tokenizer_load_seconds, model_load_seconds = _load_text_encoder_components(args, dev)
    tpu_runtime.configure_precision(args.matmul_precision)
    base_text_encoder = pipe.text_encoder

    results = {}
    warm_count = max(args.runs, 2)
    for mode_name, eager_mode, use_gdn_pallas, use_compile in mode_specs:
        tpu_runtime.reset_peak_memory_stats()
        t_h2d0 = time.perf_counter()
        base_text_encoder.to(dev)
        tpu_runtime.synchronize()
        h2d_seconds = time.perf_counter() - t_h2d0

        try:
            if use_compile:
                backend = "tpu" if dev.type == "tpu" else "inductor"
                pipe.text_encoder = torch.compile(base_text_encoder, backend=backend, dynamic=False)
            else:
                pipe.text_encoder = base_text_encoder

            with (
                tpu_runtime.configure_eager_mode(eager_mode, dev),
                tpu_qwen35.qwen35_gdn_pallas_context(
                    enabled=use_gdn_pallas,
                    model=base_text_encoder,
                    buckets=gdn_buckets,
                    require_tpu_device=(dev.type == "tpu"),
                ) as gdn_handle,
            ):
                _, pos_mask, cold_fwd, cold_d2h = _timed_encode_single(
                    pipe, args.prompt, args.max_sequence_length
                )
                neg_fwd, neg_d2h, neg_tokens = (None, None, 0)
                if args.guidance_scale != 1.0:
                    _, neg_mask, neg_fwd, neg_d2h = _timed_encode_single(
                        pipe, args.negative_prompt, args.max_sequence_length
                    )
                    neg_tokens = int(neg_mask.sum().item())

                warm_fwds = []
                warm_d2hs = []
                for _ in range(warm_count):
                    _, _, w_fwd, w_d2h = _timed_encode_single(
                        pipe, warm_prompt, args.max_sequence_length
                    )
                    warm_fwds.append(round(w_fwd, 3))
                    warm_d2hs.append(round(w_d2h, 3))

                peak_gib = tpu_runtime.max_memory_allocated_gib()
                gdn_stats = (
                    {
                        "pallas_calls": int(gdn_handle.stats["pallas_calls"]),
                        "fallback_calls": int(gdn_handle.stats["fallback_calls"]),
                        "fallback_reasons": dict(gdn_handle.stats["fallback_reasons"]),
                    }
                    if gdn_handle is not None
                    else None
                )
        finally:
            pipe.text_encoder = base_text_encoder
            base_text_encoder.to("cpu")
            gc.collect()
            tpu_runtime.empty_cache()

        results[mode_name] = {
            "eager_mode": eager_mode,
            "gdn_pallas": use_gdn_pallas,
            "compile": use_compile,
            "h2d_transfer_seconds": round(h2d_seconds, 3),
            "prompt_tokens": int(pos_mask.sum().item()),
            "negative_tokens": neg_tokens,
            "cold_encode_seconds": round(cold_fwd, 3),
            "cold_d2h_seconds": round(cold_d2h, 3),
            "neg_encode_seconds": round(neg_fwd, 3) if neg_fwd is not None else None,
            "neg_d2h_seconds": round(neg_d2h, 3) if neg_d2h is not None else None,
            "warm_prompt_distinct": warm_prompt != args.prompt,
            "warm_encode_seconds": warm_fwds,
            "warm_d2h_seconds": warm_d2hs,
            "warm_mean_seconds": round(statistics.mean(warm_fwds), 3),
            "warm_median_seconds": round(statistics.median(warm_fwds), 3),
            "peak_hbm_gib": peak_gib,
            "gdn_pallas_stats": gdn_stats,
        }

    report = {
        "mode": "encode-bench",
        "model": args.model,
        "device": tpu_runtime.device_name() or str(dev),
        "max_sequence_length": args.max_sequence_length,
        "guidance_scale": args.guidance_scale,
        "gdn_buckets": list(gdn_buckets) if gdn_buckets is not None else None,
        "tokenizer_load_seconds": round(tokenizer_load_seconds, 3),
        "model_load_seconds": round(model_load_seconds, 3),
        "results": results,
    }
    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return report


@contextmanager
def _time_denoising(pipe):
    progress_bar = pipe.progress_bar
    stats = {}

    def timed_steps(items):
        tpu_runtime.synchronize()
        started = time.perf_counter()
        count = 0
        for step in progress_bar(items):
            yield step
            count += 1
        tpu_runtime.synchronize()
        elapsed = time.perf_counter() - started
        stats.update(
            denoising_steps=count,
            denoising_seconds=round(elapsed, 3),
            ms_per_step=round(elapsed * 1000 / count, 2) if count else None,
        )

    pipe.progress_bar = timed_steps
    try:
        yield stats
    finally:
        pipe.progress_bar = progress_bar


def _run_once(
    pipe,
    args,
    embeds_payload: dict | None,
    generator_seed: int,
    output: Path,
    *,
    return_image: bool = False,
) -> dict | tuple[dict, object]:
    gen = torch.Generator("cpu").manual_seed(generator_seed)
    tpu_runtime.reset_peak_memory_stats()
    tpu_runtime.synchronize()
    t0 = time.perf_counter()

    call_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": gen,
        "max_sequence_length": args.max_sequence_length,
        "output_type": "pil",
    }
    if embeds_payload is not None:
        call_kwargs["prompt_embeds"] = embeds_payload["prompt_embeds"]
        call_kwargs["prompt_attention_mask"] = embeds_payload["prompt_attention_mask"]
        if args.guidance_scale != 1.0:
            call_kwargs["negative_prompt_embeds"] = embeds_payload["negative_prompt_embeds"]
            call_kwargs["negative_prompt_attention_mask"] = embeds_payload[
                "negative_prompt_attention_mask"
            ]
        seq_len = int(embeds_payload["prompt_embeds"].shape[1])
    else:
        call_kwargs["prompt"] = args.prompt
        call_kwargs["negative_prompt"] = args.negative_prompt
        seq_len = None

    with _time_denoising(pipe) as denoising:
        result = pipe(**call_kwargs)
    tpu_runtime.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gib = tpu_runtime.max_memory_allocated_gib()

    output.parent.mkdir(parents=True, exist_ok=True)
    image = result.images[0]
    image.save(output)
    stats = {
        "elapsed_seconds": round(elapsed, 3),
        **denoising,
        "peak_hbm_gib": round(peak_gib, 3) if peak_gib is not None else None,
        "prompt_seq_len_input": seq_len,
        "output_image": str(output.resolve()),
    }
    if return_image:
        return stats, image
    return stats


def _parse_switch_modes(raw: str) -> tuple[str, ...]:
    tokens = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = ("d2h", "drop_tpu")
    if not tokens or len(set(tokens)) != len(tokens) or any(tok not in allowed for tok in tokens):
        raise ValueError(f"--switch-modes must contain distinct choices from {allowed}, got {raw!r}.")
    return tuple(tokens)


def _summarize_stage_switches(records: list[dict]) -> dict:
    keys = (
        "pre_sync_seconds",
        "d2h_seconds",
        "drop_tpu_seconds",
        "gc_seconds",
        "empty_cache_seconds",
        "h2d_seconds",
        "total_seconds",
    )
    summary = {k: round(sum(float(rec.get(k, 0.0)) for rec in records), 4) for k in keys}
    summary["switch_count"] = len(records)
    return summary


def _image_parity_stats(ref_img, cand_img) -> dict | None:
    if not hasattr(ref_img, "tobytes") or not hasattr(cand_img, "tobytes"):
        return None
    if ref_img.size != cand_img.size or ref_img.mode != cand_img.mode:
        return None
    ref_t = torch.frombuffer(bytearray(ref_img.convert("RGB").tobytes()), dtype=torch.uint8).to(
        torch.float64
    )
    cand_t = torch.frombuffer(bytearray(cand_img.convert("RGB").tobytes()), dtype=torch.uint8).to(
        torch.float64
    )
    diff = (cand_t - ref_t).abs()
    mse = float(((cand_t - ref_t) ** 2).mean().item())
    psnr = float("inf") if mse == 0.0 else round(float(10.0 * torch.log10(torch.tensor(255.0**2 / mse)).item()), 2)
    cos = float(torch.nn.functional.cosine_similarity(ref_t.unsqueeze(0), cand_t.unsqueeze(0)).item())
    return {
        "cosine_similarity": round(cos, 6),
        "mean_abs_diff": round(float(diff.mean().item()), 4),
        "max_abs_diff": round(float(diff.max().item()), 4),
        "psnr_db": psnr,
    }


def _write_and_print_report(report: dict, save_report: str | None) -> dict:
    if save_report:
        report_path = Path(save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run_generate_or_benchmark(args) -> dict:
    dev = _require_tpu_or_override(args.allow_cpu)
    buckets = _parse_buckets(args.buckets)
    embeds_payload = (
        torch.load(args.prompt_embeds, map_location="cpu", weights_only=True)
        if args.prompt_embeds
        else None
    )
    prompt_cache_dir = getattr(args, "prompt_cache_dir", None)
    use_gdn_pallas = bool(getattr(args, "gdn_pallas", False))
    gdn_buckets = _resolve_gdn_buckets(args)
    prompt_cache_hits = None
    if embeds_payload is None and prompt_cache_dir is not None:
        has_pos = (
            tpu_qwen35.load_cached_prompt_embed(
                prompt_cache_dir, args.model, args.prompt, args.max_sequence_length
            )
            is not None
        )
        has_neg_hit = (
            False
            if args.guidance_scale == 1.0
            else (
                tpu_qwen35.load_cached_prompt_embed(
                    prompt_cache_dir, args.model, args.negative_prompt, args.max_sequence_length
                )
                is not None
            )
        )
        has_neg = args.guidance_scale == 1.0 or has_neg_hit
        include_text_encoder = not (has_pos and has_neg)
        prompt_cache_hits = {"positive": has_pos, "negative": has_neg_hit}
    else:
        include_text_encoder = embeds_payload is None
    output = Path(args.output)

    stage_switch_mode = getattr(args, "stage_switch_mode", "auto")

    if args.mode == "generate":
        text_eager_mode = _resolve_text_eager_mode(
            getattr(args, "text_eager_mode", "auto"),
            eager_mode=args.eager_mode,
            gdn_pallas=use_gdn_pallas,
        )
        t0 = time.perf_counter()
        pipe = tpu_runtime.load_pipeline(
            args.model,
            offload=args.offload,
            stage_switch_mode=stage_switch_mode,
            compile_transformer=args.compile,
            local_files_only=args.local_files_only,
            prompt_buckets=buckets,
            eager_mode=args.eager_mode,
            text_eager_mode=text_eager_mode,
            matmul_precision=args.matmul_precision,
            include_text_encoder=include_text_encoder,
            device=dev,
            prompt_cache_dir=prompt_cache_dir,
            gdn_pallas=use_gdn_pallas,
            gdn_buckets=gdn_buckets,
        )
        load_seconds = round(time.perf_counter() - t0, 2)
        try:
            runs = []
            reset_history = getattr(pipe, "reset_stage_switch_history", None)
            for idx in range(args.runs):
                run_output = output if idx == 0 else output.with_stem(f"{output.stem}-{idx + 1}")
                if callable(reset_history):
                    reset_history()
                stats = _run_once(pipe, args, embeds_payload, args.seed + idx, run_output)
                switches = getattr(pipe, "stage_switch_history", None)
                if switches:
                    stats["stage_switches"] = switches
                    stats["stage_switch_totals"] = _summarize_stage_switches(switches)
                stats["run_index"] = idx + 1
                runs.append(stats)
            gdn_stats = pipe.gdn_pallas_stats
            effective_switch_mode = getattr(pipe, "effective_stage_switch_mode", None)
        finally:
            pipe.disable_gdn_pallas()
        report = {
            "mode": "generate",
            "model": args.model,
            "device": tpu_runtime.device_name() or str(dev),
            "offload": args.offload,
            "stage_switch_mode": stage_switch_mode,
            "effective_stage_switch_mode": effective_switch_mode,
            "eager_mode": args.eager_mode,
            "text_eager_mode": text_eager_mode,
            "compile": args.compile,
            "text_encoder_loaded": include_text_encoder,
            "prompt_cache_hits": prompt_cache_hits,
            "gdn_pallas_requested": use_gdn_pallas,
            "gdn_pallas": use_gdn_pallas and include_text_encoder,
            "gdn_pallas_active": gdn_stats is not None,
            "gdn_pallas_stats": gdn_stats,
            "prompt_cache_dir": prompt_cache_dir,
            "buckets": list(buckets),
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "load_seconds": load_seconds,
            "runs": runs,
        }
    else:
        modes = [mode.strip() for mode in args.bench_modes.split(",") if mode.strip()]
        if not modes or len(set(modes)) != len(modes):
            raise ValueError("--bench-modes must contain one or more distinct modes.")
        bench_configs = []
        for item in modes:
            if item == "eager":
                bench_configs.append(("eager", "default", False))
            elif item == "defer_and_fuse":
                bench_configs.append(("defer_and_fuse", "defer_and_fuse", False))
            elif item == "compile":
                bench_configs.append(("compile", "default", True))
            else:
                raise ValueError(f"Unknown benchmark mode: {item!r}")

        results = {}
        gdn_stats_by_mode = {}
        for name, eager_mode, compile_flag in bench_configs:
            text_eager_mode = _resolve_text_eager_mode(
                getattr(args, "text_eager_mode", "auto"),
                eager_mode=eager_mode,
                gdn_pallas=use_gdn_pallas,
            )
            pipe = tpu_runtime.load_pipeline(
                args.model,
                offload=args.offload,
                stage_switch_mode=stage_switch_mode,
                compile_transformer=compile_flag,
                local_files_only=args.local_files_only,
                prompt_buckets=buckets,
                eager_mode=eager_mode,
                text_eager_mode=text_eager_mode,
                matmul_precision=args.matmul_precision,
                include_text_encoder=include_text_encoder,
                device=dev,
                prompt_cache_dir=prompt_cache_dir,
                gdn_pallas=use_gdn_pallas,
                gdn_buckets=gdn_buckets,
            )
            try:
                mode_runs = []
                reset_history = getattr(pipe, "reset_stage_switch_history", None)
                for idx in range(max(args.runs, 2)):
                    run_output = output.with_stem(f"{output.stem}-{name}-{idx + 1}")
                    if callable(reset_history):
                        reset_history()
                    stats = _run_once(pipe, args, embeds_payload, args.seed, run_output)
                    switches = getattr(pipe, "stage_switch_history", None)
                    if switches:
                        stats["stage_switches"] = switches
                        stats["stage_switch_totals"] = _summarize_stage_switches(switches)
                    stats["run_index"] = idx + 1
                    mode_runs.append(stats)
            finally:
                gdn_stats_by_mode[name] = pipe.gdn_pallas_stats
                pipe.disable_gdn_pallas()
            results[name] = mode_runs
            del pipe
            gc.collect()
            tpu_runtime.empty_cache()

        report = {
            "mode": "benchmark",
            "model": args.model,
            "device": tpu_runtime.device_name() or str(dev),
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "seed": args.seed,
            "matmul_precision": args.matmul_precision,
            "offload": args.offload,
            "stage_switch_mode": stage_switch_mode,
            "text_eager_mode": getattr(args, "text_eager_mode", "auto"),
            "text_encoder_loaded": include_text_encoder,
            "prompt_cache_hits": prompt_cache_hits,
            "gdn_pallas_requested": use_gdn_pallas,
            "gdn_pallas": use_gdn_pallas and include_text_encoder,
            "gdn_pallas_active": any(v is not None for v in gdn_stats_by_mode.values()),
            "gdn_pallas_stats_by_mode": gdn_stats_by_mode if use_gdn_pallas else None,
            "prompt_cache_dir": prompt_cache_dir,
            "buckets": list(buckets),
            "results": results,
        }

    return _write_and_print_report(report, args.save_report)


def run_stage_switch_bench(args) -> dict:
    dev = _require_tpu_or_override(args.allow_cpu)
    buckets = _parse_buckets(args.buckets)
    switch_modes = _parse_switch_modes(getattr(args, "switch_modes", "d2h,drop_tpu"))
    use_gdn_pallas = bool(getattr(args, "gdn_pallas", False))
    gdn_buckets = _resolve_gdn_buckets(args)
    text_eager_mode = _resolve_text_eager_mode(
        getattr(args, "text_eager_mode", "auto"),
        eager_mode=args.eager_mode,
        gdn_pallas=use_gdn_pallas,
    )
    prompt_1 = args.prompt
    prompt_2 = getattr(args, "warm_prompt", None) or f"{args.prompt} Warm ambient rim light."
    if prompt_2 == prompt_1:
        prompt_2 = f"{args.prompt} Alternate angle."

    output = Path(args.output)
    host_mem_init = tpu_runtime.host_memory_info_gib()
    t0 = time.perf_counter()
    pipe = tpu_runtime.load_pipeline(
        args.model,
        offload="model",
        stage_switch_mode=switch_modes[0],
        compile_transformer=False,
        local_files_only=args.local_files_only,
        prompt_buckets=buckets,
        eager_mode=args.eager_mode,
        text_eager_mode=text_eager_mode,
        matmul_precision=args.matmul_precision,
        include_text_encoder=True,
        device=dev,
        prompt_cache_dir=None,
        gdn_pallas=use_gdn_pallas,
        gdn_buckets=gdn_buckets,
    )
    load_seconds = round(time.perf_counter() - t0, 2)

    results = {}
    baseline_imgs = None
    orig_prompt = args.prompt
    try:
        for mode in switch_modes:
            pipe._activate_stage("cpu")
            pipe._prompt_cache.clear()
            pipe.set_stage_switch_mode(mode, clear_snapshots=(mode == "d2h"))
            gc.collect()
            tpu_runtime.empty_cache()

            pipe.reset_stage_switch_history()
            args.prompt = prompt_1
            run1_stats, run1_img = _run_once(
                pipe,
                args,
                None,
                args.seed,
                output.with_stem(f"{output.stem}-{mode}-p1"),
                return_image=True,
            )
            run1_switches = pipe.stage_switch_history

            pipe.reset_stage_switch_history()
            args.prompt = prompt_2
            run2_stats, run2_img = _run_once(
                pipe,
                args,
                None,
                args.seed,
                output.with_stem(f"{output.stem}-{mode}-p2"),
                return_image=True,
            )
            run2_switches = pipe.stage_switch_history

            if baseline_imgs is None:
                baseline_imgs = (mode, run1_img, run2_img)
                parity = None
            else:
                parity = {
                    "reference_mode": baseline_imgs[0],
                    "run_1": _image_parity_stats(baseline_imgs[1], run1_img),
                    "run_2": _image_parity_stats(baseline_imgs[2], run2_img),
                }

            results[mode] = {
                "effective_stage_switch_mode": pipe.effective_stage_switch_mode,
                "run_1_cold_prompt": {
                    **run1_stats,
                    "stage_switches": run1_switches,
                    "switch_totals": _summarize_stage_switches(run1_switches),
                },
                "run_2_interactive_prompt_switch": {
                    **run2_stats,
                    "stage_switches": run2_switches,
                    "switch_totals": _summarize_stage_switches(run2_switches),
                },
                "host_memory_after": tpu_runtime.host_memory_info_gib(),
                "parity_vs_baseline": parity,
            }
    finally:
        args.prompt = orig_prompt
        pipe.disable_gdn_pallas()

    comparison = None
    if "d2h" in results and "drop_tpu" in results:
        d2h_r1 = results["d2h"]["run_1_cold_prompt"]
        drop_r1 = results["drop_tpu"]["run_1_cold_prompt"]
        d2h_r2 = results["d2h"]["run_2_interactive_prompt_switch"]
        drop_r2 = results["drop_tpu"]["run_2_interactive_prompt_switch"]
        d2h_r2_sw = d2h_r2["switch_totals"]["total_seconds"]
        drop_r2_sw = drop_r2["switch_totals"]["total_seconds"]
        comparison = {
            "baseline_mode": "d2h",
            "candidate_mode": "drop_tpu",
            "run_1_switch_saved_seconds": round(
                d2h_r1["switch_totals"]["total_seconds"] - drop_r1["switch_totals"]["total_seconds"], 4
            ),
            "run_2_interactive_switch_saved_seconds": round(d2h_r2_sw - drop_r2_sw, 4),
            "run_2_interactive_switch_speedup": (
                round(d2h_r2_sw / drop_r2_sw, 3) if drop_r2_sw > 0 else None
            ),
            "run_2_e2e_saved_seconds": round(
                d2h_r2["elapsed_seconds"] - drop_r2["elapsed_seconds"], 3
            ),
        }

    report = {
        "mode": "stage-switch-bench",
        "model": args.model,
        "device": tpu_runtime.device_name() or str(dev),
        "eager_mode": args.eager_mode,
        "text_eager_mode": text_eager_mode,
        "gdn_pallas": use_gdn_pallas,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "seed": args.seed,
        "prompts": {"prompt_1": prompt_1, "prompt_2": prompt_2},
        "host_memory_before": host_mem_init,
        "load_seconds": load_seconds,
        "results": results,
        "comparison": comparison,
    }
    return _write_and_print_report(report, args.save_report)


def _resolve_text_eager_mode(text_eager_mode: str, *, eager_mode: str, gdn_pallas: bool) -> str:
    if text_eager_mode == "auto":
        return "default" if gdn_pallas else eager_mode
    return text_eager_mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MOSS-Image on Google Cloud TPU (TorchTPU).")
    parser.add_argument(
        "--mode",
        choices=("probe", "encode-only", "encode-bench", "generate", "benchmark", "stage-switch-bench"),
        default="generate",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="A studio macro photo of a ceramic teapot on a linen table.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--prompt-embeds", default=None, help="Path to precomputed .pt prompt embeddings.")
    parser.add_argument("--save-embeds", default="outputs/prompt_embeds.pt")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload", choices=tpu_runtime.VALID_OFFLOAD_MODES, default="auto")
    parser.add_argument(
        "--stage-switch-mode",
        choices=tpu_runtime.VALID_STAGE_SWITCH_MODES,
        default="auto",
        help="How staged offload leaves TPU ('drop_tpu' skips frozen-weight D2H using CPU snapshots; 'd2h' moves back to CPU).",
    )
    parser.add_argument(
        "--switch-modes",
        default="d2h,drop_tpu",
        help="Comma-separated stage switch modes for --mode stage-switch-bench (d2h, drop_tpu).",
    )
    parser.add_argument("--eager-mode", choices=tpu_runtime.VALID_EAGER_MODES, default="default")
    parser.add_argument(
        "--text-eager-mode",
        choices=("auto", *tpu_runtime.VALID_EAGER_MODES),
        default="auto",
        help="Stage-1 Qwen3.5 eager mode ('auto' uses 'default' with --gdn-pallas and --eager-mode otherwise).",
    )
    parser.add_argument("--matmul-precision", choices=tpu_runtime.VALID_MATMUL_PRECISIONS, default="high")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--gdn-pallas",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental: route Qwen3.5 GDN recurrence through deltanet-tpu Pallas kernels.",
    )
    parser.add_argument(
        "--gdn-buckets",
        default=",".join(str(b) for b in tpu_qwen35.DEFAULT_GDN_BUCKETS),
        help="Chunk-aligned (multiple of 128) static sequence buckets for GDN Pallas.",
    )
    parser.add_argument(
        "--prompt-cache-dir",
        default=None,
        help="Optional directory for caching encoded CPU prompt embeddings across runs.",
    )
    parser.add_argument(
        "--warm-prompt",
        default=None,
        help="Optional second prompt for warm distinct-text Stage-1 timing in encode-bench.",
    )
    parser.add_argument("--buckets", default="512,1024,2048,4096")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--bench-modes", default="eager,defer_and_fuse")
    parser.add_argument("--output", default="outputs/moss_tpu.png")
    parser.add_argument("--save-report", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Fallback to CPU for dry runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.mode == "probe":
        info = run_probe(args)
        if "matmul_smoke_ms" not in info:
            print("No TorchTPU device detected. Use a TPU runtime or pass --allow-cpu.", file=sys.stderr)
            return 1
    elif args.mode == "encode-only":
        run_encode_only(args)
    elif args.mode == "encode-bench":
        run_encode_bench(args)
    elif args.mode == "stage-switch-bench":
        run_stage_switch_bench(args)
    else:
        run_generate_or_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
