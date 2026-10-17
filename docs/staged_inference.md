# Separate prompt encoding and image generation

[简体中文](staged_inference_zh.md)

`moss_batch` saves prompt encodings and generates images in a separate Python process. The encoding process loads the text encoder and tokenizer; the generation process loads the transformer, scheduler, and VAE, with no text encoder or tokenizer. You can reuse encodings while changing seeds and sampling settings, or copy the encoding directory to another machine.

The existing pipeline already supports CPU offload. This workflow avoids repeated text encoding and loading the text encoder during generation; memory usage and runtime depend on your hardware, settings, and offload mode. It does not change the model or sampling algorithm.

## Run the two stages

From the repository root, install the package in your inference environment:

```sh
python -m pip install -e .
python -m moss_batch validate --jobs examples/staged_jobs.jsonl
```

`validate` checks only the manifest and does not import inference dependencies or download a model. The sample has two jobs with identical prompts and different seeds/CFG values; encoding is computed once and saved under both job IDs.

Choose a model commit from its Hugging Face history and replace `MODEL_COMMIT_SHA` below. Use the same model and commit for both commands:

```sh
python -m moss_batch encode --jobs examples/staged_jobs.jsonl --model OpenMOSS-Team/MOSS-Image-Base --revision MODEL_COMMIT_SHA --cache-dir encodings
python -m moss_batch generate --jobs examples/staged_jobs.jsonl --model OpenMOSS-Team/MOSS-Image-Base --revision MODEL_COMMIT_SHA --cache-dir encodings --output-dir outputs
```

Wait for `encode` to exit before running `generate`. These are separate process invocations, so encoding-stage model allocations are released before generation starts. For Hub models, each stage downloads only its required component directories and pipeline code. If `--revision` is omitted, the command resolves the current Hub revision and prints it; pinning a commit prevents an upstream update between stages from invalidating the cache.

CUDA with `--offload model` is the default. `--offload sequential` uses sequential CPU offload; `--offload none` keeps the stage's components on the selected device. CPU execution requires both `--device cpu --offload none`. Add `--local-files-only` to use already cached Hub files without downloading missing components.

## Task manifest

The manifest is UTF-8 JSONL: one flat JSON object per line and one output image per job. Blank lines are allowed. Required fields are `id` and `prompt`; unknown fields and duplicate IDs are rejected.

| Field | Default | Requirements |
| --- | --- | --- |
| `id` | Required | 1–128 ASCII letters, digits, `_`, or `-`; start with a letter/digit; unique ignoring case; no reserved Windows device names |
| `prompt` | Required | Nonempty string |
| `negative_prompt` | `""` | String |
| `seed` | `-1` | `-1` chooses and saves a random seed; otherwise integer `0` through `2**53 - 1` |
| `height`, `width` | `1024` | Positive multiples of 16 |
| `num_inference_steps` | `50` | Positive integer |
| `guidance_scale` | `3.0` | Finite, nonnegative number |
| `shift` | `5.0` | Finite, positive number |
| `max_sequence_length` | `4096` | Integer from 1 to 4096 |
| `uncond_skip_layers` | `[22]` | Array of unique nonnegative layer indices; `[]` disables this skipping |
| `slg_layers` | `[]` | Array of unique nonnegative layer indices |
| `slg_scale` | `0.0` | Finite, nonnegative number |
| `slg_start`, `slg_stop` | `0.0`, `1.0` | `0 <= slg_start <= slg_stop <= 1` |

Layer indices are checked against the loaded transformer's depth before generation. Prompts, including Unicode, whitespace, and any Layout JSON represented as a string, are preserved exactly; no prompt prefix or enhancement is applied. Both positive and negative branches are encoded even when `guidance_scale` is 1, allowing later CFG changes.

These field names follow the pipeline (`num_inference_steps`, `guidance_scale`), not Gradio's `steps`/`cfg` names. Gradio parameter files are not valid batch manifests.

## Reuse and move encodings

Each `encodings/<id>/` contains `encoding.safetensors` and `metadata.json`. Metadata records the source identity, exact prompts, maximum sequence length, dtype, encoder/tokenizer classes, chat template, and encoding dependency versions. Tensor shape, dtype, attention masks, and finite values are validated when loading.

To try another seed, CFG, step count, shift, or image size, keep the job ID and encoding-related fields unchanged, edit the sampling fields, and run `generate` with a **new `--output-dir`**. Encoding can be reused. Changing the prompt, negative prompt, maximum sequence length, or model source requires a new encoding; use a new cache directory or a new job ID and run `encode` again. Existing incompatible caches are never overwritten.

To generate on another machine, copy the complete encoding directory and manifest, install the package there, and use the same pinned Hub model identity. Generation downloads the transformer, scheduler, VAE, and pipeline code, without downloading text encoder weights.

For local model bundles, use a portable version label instead of `--revision`:

```sh
python -m moss_batch encode --jobs examples/staged_jobs.jsonl --model /path/to/model --local-model-version my-bundle-v1 --cache-dir encodings
python -m moss_batch generate --jobs examples/staged_jobs.jsonl --model /path/to/model --local-model-version my-bundle-v1 --cache-dir encodings --output-dir outputs
```

The generation bundle needs the official pipeline code, model index, transformer, scheduler, and VAE. Local paths may differ across machines, but the same version label must identify the same source bundle. Change the label whenever model, tokenizer, or pipeline files change. This label is supplied by you; the tool does not verify that two bundles with the same label contain identical files. A local-bundle identity and a Hub identity are distinct even if the files came from the same model.

## Resume an interrupted batch

Run the same `generate` command again. Before inference starts for a job, `outputs/<id>/parameters.json` records its actual seed and settings. `01.png` is published atomically, then the record is marked complete. A completed job is skipped only when settings and provenance match and its PNG fully decodes at the requested dimensions. Missing or damaged images are regenerated with the saved seed. Changed settings or provenance are rejected; use a new output directory for a new run.

The output record uses this tool's schema version 1 and includes the job, encoding/model provenance, actual seed, completion status, and image filename. It is separate from Gradio's parameter schema; metadata is stored in the JSON sidecar, not embedded in the PNG. Retaining a seed supports replay, but does not guarantee pixel-identical output across hardware or software versions.

Each command locks its destination root with `.moss-batch.lock`. Normal exit, handled failures, and Ctrl+C release the lock. After a hard process termination, check that no writer is running before manually removing a stale lock. If a cache is corrupt or incompatible, use a new cache path, or remove only the affected job's cache directory after checking it and re-run `encode`. The tool does not delete or repair existing caches automatically.
