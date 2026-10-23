"""MOSS-Image Gradio interface."""

import argparse
import gc
import hashlib
import html
import importlib.metadata
import inspect
import json
import logging
import math
import secrets
import threading
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from importlib.resources import files
from pathlib import Path
from secrets import randbits
from uuid import uuid4

import torch

from gradio_presets import (
    PIXELS,
    fit_size,
    pixel_label,
    ratio_value,
    size_selection,
)
from inference_runtime import (
    DEFAULT_MODEL,
    configure_pipeline,
    cuda_budget_gib,
    generate_with_progress,
    model_source,
    set_scheduler,
)
from inference_runtime import (
    load_pipeline as load_hf_pipeline,
)

logger = logging.getLogger(__name__)

COMPONENTS = ("transformer", "text_encoder", "tokenizer", "vae", "scheduler")
PROFILES = ("auto", "minimal", "16gb", "24gb", "48gb")


def is_model_bundle(path):
    try:
        if not (path / "pipeline_moss_image.py").is_file():
            return False
        index = json.loads((path / "model_index.json").read_text())
        if not all(name in index and (path / name).is_dir() for name in COMPONENTS):
            return False
        for name in ("transformer", "text_encoder", "vae"):
            component = path / name
            if not (component / "config.json").is_file():
                return False
            indices = list(component.glob("*.safetensors.index.json"))
            if indices:
                for file in indices:
                    shards = json.loads(file.read_text())["weight_map"].values()
                    if not shards or not all((component / shard).is_file() for shard in shards):
                        return False
            elif not any(component.glob("*.safetensors")):
                return False
        tokenizer = path / "tokenizer"
        return (
            (path / "scheduler" / "scheduler_config.json").is_file()
            and (tokenizer / "tokenizer_config.json").is_file()
            and (
                (tokenizer / "tokenizer.json").is_file()
                or ((tokenizer / "vocab.json").is_file() and (tokenizer / "merges.txt").is_file())
            )
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def discover_models(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return []
    candidates = [root] if is_model_bundle(root) else sorted(p for p in root.iterdir() if p.is_dir())
    return sorted({str(p.resolve()) for p in candidates if is_model_bundle(p)})


def profile_defaults(profile, vram_gb=None):
    if profile == "auto":
        if vram_gb is None:
            vram_gb = cuda_budget_gib() if torch.cuda.is_available() else 24
        profile = (
            "minimal" if vram_gb < 12 else "16gb" if vram_gb < 20 else "24gb" if vram_gb < 40 else "48gb"
        )
    return {
        "minimal": ("hybrid", 1024, 1024),
        "16gb": ("model", 1024, 1024),
        "24gb": ("model", 1024, 1024),
        "48gb": ("none", 1024, 1024),
    }[profile]


@dataclass(frozen=True)
class GenerationConfig:
    prompt: str = ""
    negative_prompt: str = ""
    height: int = 1024
    width: int = 1024
    steps: int = 50
    cfg: float = 3.0
    shift: float = 5.0
    seed: int = 42
    num_images: int = 1
    max_sequence_length: int = 4096
    uncond_skip_layers: str = "22"
    slg_layers: str = ""
    slg_scale: float = 0.0
    slg_start: float = 0.0
    slg_stop: float = 1.0
    offload: str = "model"
    compile_transformer: bool = False

    def layers(self, name):
        value = getattr(self, name)
        if not value.strip():
            return ()
        try:
            layers = tuple(int(item.strip()) for item in value.split(","))
        except ValueError as exc:
            raise ValueError("SLG layers must be comma-separated integer indices, e.g. 22.") from exc
        if any(layer < 0 for layer in layers) or len(set(layers)) != len(layers):
            raise ValueError("SLG indices must be nonnegative and unique.")
        return layers

    def validate(self):
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Enter a prompt.")
        for name in ("height", "width", "steps", "seed", "num_images", "max_sequence_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer.")
        if self.height <= 0 or self.width <= 0 or self.height % 16 or self.width % 16:
            raise ValueError("Height and width must be positive multiples of 16.")
        if self.steps < 1 or self.num_images < 1:
            raise ValueError("Steps and image count must be positive.")
        if not math.isfinite(self.cfg) or self.cfg < 0 or not math.isfinite(self.shift) or self.shift <= 0:
            raise ValueError("CFG must be finite and nonnegative; shift must be finite and positive.")
        if not -1 <= self.seed < 2**53:
            raise ValueError("Seed must be -1 (random) or an integer from 0 to 2⁵³−1 (browser-safe range).")
        if not 1 <= self.max_sequence_length <= 4096:
            raise ValueError("Maximum text length must be between 1 and 4096.")
        if self.offload not in ("none", "model", "sequential", "hybrid"):
            raise ValueError("Unknown offload mode.")
        if not isinstance(self.compile_transformer, bool):
            raise ValueError("Compile transformer must be true or false.")
        if self.offload in ("sequential", "hybrid") and self.compile_transformer:
            raise ValueError("Compile transformer is incompatible with sequential or low-memory offload.")
        if not math.isfinite(self.slg_scale) or self.slg_scale < 0:
            raise ValueError("SLG scale must be finite and nonnegative.")
        if not 0 <= self.slg_start <= self.slg_stop <= 1:
            raise ValueError("Require 0 <= SLG start <= SLG stop <= 1.")
        self.layers("uncond_skip_layers")
        self.layers("slg_layers")

    def pipeline_kwargs(self):
        return dict(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=self.steps,
            guidance_scale=self.cfg,
            num_images_per_prompt=self.num_images,
            max_sequence_length=self.max_sequence_length,
            uncond_skip_layers=self.layers("uncond_skip_layers"),
            slg_layers=self.layers("slg_layers"),
            slg_scale=self.slg_scale,
            slg_start=self.slg_start,
            slg_stop=self.slg_stop,
        )


def read_prompt_defaults(path):
    headers = {"[prompt_prefix]": "prompt_prefix", "[negative_prompt]": "negative_prompt"}
    sections = {}
    current = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        header = line.strip()
        if header in headers:
            current = headers[header]
            if current in sections:
                raise ValueError(f"Duplicate section {header} in {path}.")
            sections[current] = []
            continue
        if current is None:
            if header and not header.startswith("#"):
                raise ValueError(f"Expected [prompt_prefix] or [negative_prompt] in {path}.")
            continue
        sections[current].append(line)
    if sections.keys() != set(headers.values()):
        raise ValueError(f"Both [prompt_prefix] and [negative_prompt] are required in {path}.")
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def parse_parameters(parameters, version):
    if version not in (1, 2):
        raise ValueError("Unsupported generation metadata version.")
    parameters = dict(parameters)
    if version == 1:
        # Version 1 called negative-branch skipping "slg_layers".
        layers = parameters.pop("slg_layers", "")
        parameters["uncond_skip_layers"] = layers if parameters.pop("slg", True) else ""
    if parameters.get("seed") is None:
        parameters["seed"] = -1
    parameters.setdefault("compile_transformer", False)
    if parameters.get("offload") == "staged":
        parameters["offload"] = "model"
    config = GenerationConfig(**parameters)
    config.validate()
    return config


def load_pipeline(model, offload, compile_transformer=False, local_files_only=False):
    if not torch.cuda.is_available():
        raise ValueError("A CUDA GPU is required for generation. The interface can be previewed without one.")
    pipe = load_hf_pipeline(model, local_files_only=local_files_only)
    return configure_pipeline(pipe, offload=offload, compile_transformer=compile_transformer)


class ModelRunner:
    def __init__(self, output_dir, local_files_only):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.local_files_only = local_files_only
        self.pipe = None
        self.key = None
        self.lock = threading.RLock()

    def unload(self):
        with self.lock:
            if self.pipe is not None:
                self.pipe.remove_all_hooks()
            self.pipe = None
            self.key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return "Model released. It will load again on the next generation."

    def generate(self, model, config, progress=None):
        config.validate()
        model = model_source(model)
        if Path(model).is_dir() and not is_model_bundle(Path(model)):
            raise ValueError("Choose a complete local model bundle with all five components.")
        with self.lock:
            config = replace(config, seed=secrets.randbelow(2**32)) if config.seed == -1 else config
            start = time.monotonic()
            total_steps = config.steps

            def report(value, **kwargs):
                if progress is not None:
                    progress(value, **kwargs)

            def report_stage(stage):
                nonlocal total_steps
                if stage == "denoise":
                    timesteps = getattr(getattr(self.pipe, "scheduler", None), "timesteps", None)
                    if timesteps is not None and len(timesteps):
                        total_steps = len(timesteps)
                descriptions = {
                    "encode": "Encoding prompt",
                    "denoise": f"Sampling · step 0/{total_steps}",
                    "decode": "Decoding image",
                }
                report((0, total_steps) if stage == "denoise" else 0, desc=descriptions[stage], unit="steps")

            def report_step(pipe, step_index, timestep, callback_kwargs):
                completed = step_index + 1
                report(
                    (completed, total_steps), desc=f"Sampling · step {completed}/{total_steps}", unit="steps"
                )
                return callback_kwargs

            try:
                key = (str(model), config.offload, config.compile_transformer)
                if key != self.key:
                    report(0, desc="Loading model")
                    self.unload()
                    self.pipe = load_pipeline(
                        model, config.offload, config.compile_transformer, self.local_files_only
                    )
                    self.key = key
                depth = self.pipe.transformer.config.num_layers
                layers = config.layers("uncond_skip_layers") + config.layers("slg_layers")
                if any(layer >= depth for layer in layers):
                    raise ValueError(f"SLG layer index must be below {depth}.")
                set_scheduler(self.pipe, config.shift)
                images = generate_with_progress(
                    self.pipe, report_stage, report_step,
                    **config.pipeline_kwargs(),
                    generator=torch.Generator("cpu").manual_seed(config.seed),
                ).images
            except torch.cuda.OutOfMemoryError as exc:
                self.unload()
                raise ValueError(
                    "GPU memory exhausted. Try fewer/smaller images or a lower-memory offload mode."
                ) from exc
            report(0, desc="Saving images and parameters")
            folder = self.output_dir / uuid4().hex
            folder.mkdir(parents=True, exist_ok=True)
            paths = []
            for i, image in enumerate(images):
                path = folder / f"{i + 1:02d}.png"
                image.save(path)
                paths.append(str(path))
            root = Path(model)
            configs = []
            if root.is_dir():
                configs = (
                    [root / "model_index.json", root / "scheduler" / "scheduler_config.json"]
                    + [root / c / "config.json" for c in ("transformer", "text_encoder", "vae")]
                    + sorted(
                        p for p in (root / "tokenizer").iterdir() if p.suffix in (".json", ".jinja", ".txt")
                    )
                )
            record = {
                "schema_version": 2,
                "model": root.name if root.is_dir() else model,
                "pipeline_class": type(self.pipe).__name__,
                "pipeline_module": type(self.pipe).__module__,
                "parameters": asdict(config),
                "sampler": "Euler",
                "weight_dtype": "bfloat16",
                "vae_dtype": "float32",
                "pipeline_sha256": hashlib.sha256(
                    Path(inspect.getfile(type(self.pipe))).read_bytes()
                ).hexdigest(),
                "model_config_sha256": {
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in configs
                    if p.is_file()
                },
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "diffusers", "transformers", "accelerate")
                },
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "elapsed_seconds": round(time.monotonic() - start, 3),
                "images": [Path(p).name for p in paths],
            }
            metadata = folder / "parameters.json"
            metadata.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            report(1, desc="Complete")
            return (
                paths,
                str(metadata),
                f"Saved {len(images)} image(s) · seed {config.seed} · {record['elapsed_seconds']:.1f}s",
            )


UI_ASSETS = Path(str(files("assets")))
EXAMPLES = UI_ASSETS / "images"
RATIO_PRESETS = ("9:16", "2:3", "3:4", "1:1", "4:3", "3:2", "16:9")


def build_app(
    model_root,
    profile="auto",
    model=None,
    output_dir="outputs/gradio",
    prompt_defaults=UI_ASSETS / "prompt_defaults.txt",
    local_files_only=False,
):
    import gradio as gr

    runner = ModelRunner(output_dir, local_files_only)
    choices = [DEFAULT_MODEL, *discover_models(model_root)]
    if model:
        model = model_source(model)
        if model not in choices:
            choices.append(model)
    initial = model or (choices[0] if choices else None)
    offload_default, height_default, width_default = profile_defaults(profile)
    pixels_default, ratio_default = size_selection(width_default, height_default)
    min_budget, max_budget = PIXELS[0] / 1048576, PIXELS[-1] / 1048576
    defaults = GenerationConfig()
    texts = read_prompt_defaults(prompt_defaults)
    def asset_snapshot():
        paths = sorted(EXAMPLES.glob("*/parameters.json"))
        images = []
        stamps = []
        for path in paths:
            record = json.loads(path.read_text())
            image = (path.parent / record.get("image", "image.webp")).resolve()
            if not image.is_relative_to(EXAMPLES.resolve()):
                raise ValueError("Example images must be inside the assets directory.")
            images.append(str(image))
            stamps.append([str(path), path.stat().st_mtime_ns, image.stat().st_mtime_ns])
        return {"paths": [str(p) for p in paths], "images": images, "stamps": stamps,
                "defaults": Path(prompt_defaults).stat().st_mtime_ns, "selected": None}

    initial_assets = asset_snapshot()
    with gr.Blocks(title="MOSS-Image", analytics_enabled=False) as app:
        logo = app.serve_static_file(UI_ASSETS / "logo.svg")
        modelscope_base = app.serve_static_file(UI_ASSETS / "modelscope-base-badge.svg")
        modelscope_sft = app.serve_static_file(UI_ASSETS / "modelscope-sft-badge.svg")
        feishu_badge = app.serve_static_file(UI_ASSETS / "feishu-builders-badge.svg")
        gr.HTML(
            '<header style="text-align:center;padding:20px 0 24px">'
            '<div style="display:flex;align-items:center;justify-content:center;gap:20px">'
            f'<img src="{html.escape(logo["url"].lstrip("/"), quote=True)}" alt="MOSS-Image" '
            'width="56" height="56" style="display:block;object-fit:contain">'
            '<span aria-hidden="true" style="font-size:32px;font-weight:300;opacity:.3">|</span>'
            '<h1 style="margin:0;font-size:clamp(24px,3vw,34px);font-weight:650;'
            'letter-spacing:.04em;line-height:1.2">MOSS-Image</h1></div>'
            '<nav aria-label="Project links" style="display:flex;justify-content:center;'
            'align-items:center;flex-wrap:wrap;gap:8px 24px;margin-top:14px;'
            'font-size:14px;line-height:1.8;font-weight:500">'
            '<a style="color:var(--body-text-color-subdued);text-decoration:none" '
            'href="https://github.com/OpenMOSS/MOSS-Image">GitHub</a>'
            '<a style="color:var(--body-text-color-subdued);text-decoration:none" '
            'href="https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base">Base weights</a>'
            '<a target="_blank" rel="noopener noreferrer" href="https://modelscope.cn/models/openmoss/MOSS-Image-Base">'
            f'<img alt="ModelScope · MOSS-Image-Base" src="{html.escape(modelscope_base["url"].lstrip("/"), quote=True)}"></a>'
            '<a target="_blank" rel="noopener noreferrer" href="https://modelscope.cn/models/openmoss/MOSS-Image-SFT">'
            f'<img alt="ModelScope · MOSS-Image-SFT" src="{html.escape(modelscope_sft["url"].lstrip("/"), quote=True)}"></a>'
            '<a target="_blank" rel="noopener noreferrer" href="https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=7a3m2007-815f-409b-9c3d-a4129f5c058e">'
            f'<img alt="MOSS-Builders Feishu Community" src="{html.escape(feishu_badge["url"].lstrip("/"), quote=True)}"></a>'
            '<a style="color:var(--body-text-color-subdued);text-decoration:none" '
            'href="https://github.com/OpenMOSS/MOSS-Image/blob/main/README_ZH.md">文档</a>'
            '<a style="color:var(--body-text-color-subdued);text-decoration:none" '
            'href="https://x.com/Open_MOSS">OpenMOSS</a></nav></header>'
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=3, min_width=640):
                with gr.Row(equal_height=False, scale=0, elem_id="model-toolbar"):
                    checkpoint = gr.Dropdown(
                        choices=[(Path(p).name, p) for p in choices],
                        value=initial,
                        allow_custom_value=True,
                        label="Hugging Face model ID or local directory",
                        scale=3,
                        min_width=200,
                    )
                with gr.Group():
                    prompt = gr.Textbox(
                        value=texts["prompt_prefix"],
                        label="Prompt",
                        lines=4,
                        max_lines=8,
                        placeholder="Describe your image — subject, light, composition, style…",
                    )
                    negative = gr.Textbox(
                        value=texts["negative_prompt"],
                        label="Negative Prompt",
                        lines=1,
                        max_lines=3,
                        placeholder="What would you like to leave out?",
                    )
                with gr.Column():
                    resolution = gr.State(pixels_default)
                    ratio = gr.State(ratio_default)
                    with gr.Row():
                        resolution_slider = gr.Slider(
                            minimum=min_budget,
                            maximum=max_budget,
                            step=1 / 16,
                            value=pixels_default / 1048576,
                            label="Resolution (M pixels)",
                            scale=3,
                            min_width=170,
                        )
                        aspect_slider = gr.Slider(
                            minimum=0.125,
                            maximum=8,
                            step=0.001,
                            value=width_default / height_default,
                            label="Aspect ratio (W/H)",
                            scale=3,
                            min_width=170,
                        )
                        width = gr.Number(
                            value=width_default,
                            precision=0,
                            label="Width",
                            minimum=16,
                            step=16,
                            scale=1,
                            min_width=80,
                        )
                        height = gr.Number(
                            value=height_default,
                            precision=0,
                            label="Height",
                            minimum=16,
                            step=16,
                            scale=1,
                            min_width=80,
                        )
                    with gr.Row():
                        resolution_choices = gr.Radio(
                            choices=[(pixel_label(n), n) for n in PIXELS],
                            value=pixels_default,
                            label="Resolution presets",
                            scale=5,
                            min_width=240,
                        )
                        aspect_choices = gr.Radio(
                            choices=list(RATIO_PRESETS),
                            value=ratio_default if ratio_default in RATIO_PRESETS else None,
                            label="Aspect ratio presets",
                            scale=7,
                            min_width=320,
                        )
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=100):
                            count = gr.Number(value=1, precision=0, label="Images", minimum=1)
                        with gr.Column(scale=1, min_width=100):
                            random_seed = gr.Button("Random", size="sm", render=False)
                            seed = gr.Number(
                                value=42,
                                precision=0,
                                label="Seed",
                                minimum=-1,
                                maximum=2**53 - 1,
                                buttons=[random_seed],
                            )
                        with gr.Column(scale=1, min_width=100):
                            steps = gr.Number(value=defaults.steps, precision=0, minimum=1, label="Steps")
                    with gr.Accordion("Advanced settings", open=False):
                        with gr.Row():
                            cfg = gr.Number(value=defaults.cfg, label="CFG", minimum=0)
                            shift = gr.Number(value=defaults.shift, label="Shift", minimum=0.01)
                        with gr.Row():
                            offload = gr.Dropdown(
                                choices=[
                                    ("GPU resident", "none"),
                                    ("Component", "model"),
                                    ("Sequential · fallback", "sequential"),
                                    ("Low memory", "hybrid"),
                                ],
                                value=offload_default,
                                label="Memory mode",
                                scale=2,
                                min_width=140,
                            )
                            compile_transformer = gr.Checkbox(
                                value=False,
                                interactive=offload_default not in ("sequential", "hybrid"),
                                label="Compile DiT",
                                scale=0,
                                min_width=0,
                            )

                        skip_mode = gr.Radio(
                            choices=[("Off", "off"), ("Negative-branch skipping", "negative"),
                                     ("Standard SLG", "standard")],
                            value="negative",
                            label="Skip layer mode",
                        )
                        layers = gr.Textbox(
                            value=defaults.uncond_skip_layers,
                            label="Skip layers",
                            info="Zero-based layer indices, separated by commas. Empty disables skipping.",
                        )
                        with gr.Row(visible=False) as standard_settings:
                            slg_layers = gr.Textbox(
                                value=defaults.slg_layers, label="SLG layers",
                                info="Zero-based layer indices, separated by commas.",
                            )
                            slg_scale = gr.Number(value=defaults.slg_scale, minimum=0, label="SLG scale")
                        with gr.Row() as skip_interval:
                            slg_start = gr.Number(value=defaults.slg_start, minimum=0, maximum=1, label="Skip start")
                            slg_stop = gr.Number(value=defaults.slg_stop, minimum=0, maximum=1, label="Skip stop")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="生成结果", columns=2, scale=1, object_fit="contain")
                status = gr.Textbox(label="Status", interactive=False, scale=0)
        with gr.Row():
            with gr.Column(scale=3, min_width=640):
                generate = gr.Button("Generate", variant="primary")
            gr.Column(scale=2)
        gr.Markdown("## Examples")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=360):
                examples = gr.Gallery(
                    value=initial_assets["images"],
                    label="Examples",
                    columns=4,
                    object_fit="contain",
                    interactive=False,
                )
            with gr.Column(scale=2, min_width=300):
                example_prompt = gr.Textbox(
                    label="Original Prompt",
                    lines=5,
                    max_lines=8,
                    interactive=False,
                    placeholder="Click an image to view its original prompt.",
                )
                example_negative = gr.Textbox(
                    label="Negative Prompt",
                    lines=1,
                    max_lines=3,
                    interactive=False,
                )
                example_settings = gr.Textbox(
                    label="Original settings",
                    lines=2,
                    max_lines=3,
                    interactive=False,
                )
        fields = [
            prompt,
            negative,
            height,
            width,
            steps,
            cfg,
            shift,
            seed,
            count,
            layers,
            slg_layers,
            slg_scale,
            slg_start,
            slg_stop,
            offload,
            compile_transformer,
        ]
        field_names = (
            "prompt",
            "negative_prompt",
            "height",
            "width",
            "steps",
            "cfg",
            "shift",
            "seed",
            "num_images",
            "uncond_skip_layers",
            "slg_layers",
            "slg_scale",
            "slg_start",
            "slg_stop",
            "offload",
            "compile_transformer",
        )

        def run(selected, progress=gr.Progress(), *values):
            try:
                parameters = dict(zip(field_names, values))
                if parameters.get("negative_prompt") is None:
                    parameters["negative_prompt"] = ""
                config = GenerationConfig(**parameters)
                images, _, message = runner.generate(selected or "", config, progress=progress)
                return images, message
            except (ValueError, OSError) as exc:
                logger.exception("Generation failed")
                raise gr.Error(str(exc)) from exc

        def load_example(path):
            path = Path(path)
            record = json.loads(path.read_text())
            # Examples supply image settings; keep the user's runtime configuration.
            parameters = {
                key: value
                for key, value in record["parameters"].items()
                if key not in ("offload", "compile_transformer")
            }
            config = parse_parameters(parameters, record.get("schema_version", 1))
            # Runtime options stay as the user set them; seed belongs to the example.
            return (
                config.prompt,
                config.negative_prompt,
                config.height,
                config.width,
                config.steps,
                config.cfg,
                config.shift,
                config.num_images,
                config.seed,
                gr.update(value=config.uncond_skip_layers, visible=bool(config.uncond_skip_layers)),
                config.slg_layers,
                config.slg_scale,
                config.slg_start,
                config.slg_stop,
                "standard" if config.slg_layers and config.slg_scale > 0 else
                "negative" if config.uncond_skip_layers else "off",
                gr.update(visible=bool(config.slg_layers and config.slg_scale > 0)),
                gr.update(visible=bool(config.uncond_skip_layers or config.slg_layers and config.slg_scale > 0)),
                config.prompt,
                config.negative_prompt,
                (
                    f"{config.width} × {config.height} · {config.steps} steps · CFG {config.cfg:g}\n"
                    f"Euler · Seed {parameters.get('seed') if parameters.get('seed') is not None else 'unknown (random on generation)'}\n"
                    f"Shift {config.shift:g} · "
                    f"Uncond skip {config.uncond_skip_layers or 'off'} · "
                    f"Standard SLG {config.slg_layers or 'off'} (scale {config.slg_scale:g}) · "
                    f"Interval [{config.slg_start:g}, {config.slg_stop:g})"
                ),
            )

        assets = gr.State(initial_assets)

        def choose_example(state, evt: gr.SelectData):
            if evt.index < 0 or evt.index >= len(state["paths"]):
                raise gr.Error("Choose an example image.")
            path = state["paths"][evt.index]
            return (*load_example(path), {**state, "selected": path})

        def refresh_assets(state):
            fresh = asset_snapshot()
            changed = fresh["paths"] != state["paths"] or fresh["stamps"] != state["stamps"]
            selected = state["selected"]
            fresh["selected"] = selected if selected in fresh["paths"] else None
            values = [gr.skip() for _ in example_fields]
            if fresh["selected"]:
                before = state["stamps"][state["paths"].index(selected)]
                after = fresh["stamps"][fresh["paths"].index(selected)]
                if before != after:
                    values = list(load_example(selected))
            if selected and not fresh["selected"]:
                values[-3:] = ["", "", "Example removed from assets."]
            if fresh["defaults"] != state["defaults"]:
                defaults = read_prompt_defaults(prompt_defaults)
                values[0] = defaults["prompt_prefix"]
                values[1] = defaults["negative_prompt"]
            selection = fresh["paths"].index(selected) if fresh["selected"] else None
            gallery = gr.update(value=fresh["images"], selected_index=selection) if changed else gr.skip()
            return (gallery, fresh, *values)

        def update_skip(mode):
            negative = mode == "negative"
            standard = mode == "standard"
            return (
                gr.update(value="22" if negative else "", visible=negative),
                "", 3.0 if standard else 0.0, 0.01 if standard else 0.0, 1.0,
                gr.update(visible=standard), gr.update(visible=mode != "off"),
            )

        def update_compile(mode):
            if mode in ("sequential", "hybrid"):
                return gr.update(value=False, interactive=False)
            return gr.update(interactive=True)

        def apply_ratio(selected, pixels, w, h):
            if selected == "custom" or (not pixels and (not w or not h)):
                return gr.skip(), gr.skip()
            return fit_size(pixels or w * h, selected)

        def apply_resolution(pixels, selected, w, h):
            if not pixels or (selected == "custom" and (not w or not h)):
                return gr.skip(), gr.skip()
            return fit_size(pixels, f"{w}:{h}" if selected == "custom" else selected)

        def apply_budget(value, selected, w, h):
            return apply_resolution(value * 1048576, selected, w, h)

        def apply_aspect(value, pixels, w, h):
            return apply_ratio(f"{value}:1", pixels, w, h)

        def sync_size(w, h):
            pixels, selected = size_selection(w, h)
            choices = [pixels, selected, pixels or None, selected if selected in RATIO_PRESETS else None]
            if not w or not h or w <= 0 or h <= 0:
                return choices + [gr.skip(), gr.skip()]
            budget = (pixels or w * h) / 1048576
            aspect = ratio_value(selected) if selected != "custom" else w / h
            return choices + [min(max_budget, max(min_budget, budget)), min(8, max(0.125, aspect))]

        resolution_slider.release(
            apply_budget, [resolution_slider, ratio, width, height], [width, height], queue=False
        )
        aspect_slider.release(
            apply_aspect, [aspect_slider, resolution, width, height], [width, height], queue=False
        )
        resolution_choices.input(
            apply_resolution,
            [resolution_choices, ratio, width, height],
            [width, height],
            queue=False,
        )
        aspect_choices.input(
            apply_ratio,
            [aspect_choices, resolution, width, height],
            [width, height],
            queue=False,
        )
        for field in (width, height):
            field.change(
                sync_size,
                [width, height],
                [
                    resolution,
                    ratio,
                    resolution_choices,
                    aspect_choices,
                    resolution_slider,
                    aspect_slider,
                ],
                queue=False,
                api_name=False,
            )
        random_seed.click(partial(randbits, 32), outputs=seed, queue=False, api_name=False)

        example_fields = [
            prompt, negative, height, width, steps, cfg, shift, count, seed, layers,
            slg_layers, slg_scale, slg_start, slg_stop,
            skip_mode, standard_settings, skip_interval,
            example_prompt, example_negative, example_settings,
        ]
        examples.select(
            choose_example, inputs=assets, outputs=[*example_fields, assets],
            queue=False, api_name=False,
        )
        gr.Timer(2).tick(
            refresh_assets, inputs=assets, outputs=[examples, assets, *example_fields],
            queue=False, api_name=False,
        )
        skip_mode.input(
            update_skip, inputs=skip_mode,
            outputs=[layers, slg_layers, slg_scale, slg_start, slg_stop, standard_settings, skip_interval],
            queue=False, api_name=False,
        )
        offload.change(
            update_compile,
            offload,
            compile_transformer,
            queue=False,
            api_name=False,
        )
        generate.click(
            run,
            [checkpoint] + fields,
            [gallery, status],
            concurrency_limit=1,
            concurrency_id="model",
            api_name="generate",
        )
    return app.queue(default_concurrency_limit=1, max_size=8)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="auto")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(".."),
        help="Directory containing complete local model bundles",
    )
    parser.add_argument("--model", help="Hugging Face model ID or local directory")
    parser.add_argument("--local-files-only", action="store_true", help="Use only local files or cached HF files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gradio"))
    parser.add_argument(
        "--prompt-defaults",
        type=Path,
        default=UI_ASSETS / "prompt_defaults.txt",
        help="UTF-8 text file containing [prompt_prefix] and [negative_prompt] sections",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = build_app(
        args.model_root, args.profile, args.model, args.output_dir, args.prompt_defaults, args.local_files_only
    )
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        allowed_paths=[
            str(args.output_dir.expanduser().resolve()),
            str(EXAMPLES.resolve()),
            str(UI_ASSETS.resolve()),
        ],
    )


if __name__ == "__main__":
    main()
