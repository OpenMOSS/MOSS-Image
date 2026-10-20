import torch
from comfy import (
    conds,
    latent_formats,
    model_base,
    model_detection,
    model_management,
    model_patcher,
    sd,
    storage,
    utils,
)
from comfy.supported_models_base import BASE, ClipTarget
from comfy.text_encoders import hunyuan_video

from .model import MOSSImageTransformer
from .quant import unfuse_block_qkv
from .text_encoder import MOSSImageTokenizer, te

DTYPES = {
    "default": None,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e4m3fn_fast": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


class MOSSImage(model_base.BaseModel):
    def __init__(self, config, device):
        super().__init__(config, model_base.ModelType.FLOW, device=device, unet_model=MOSSImageTransformer)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        for source, target in (("cross_attn", "c_crossattn"), ("attention_mask", "attention_mask")):
            value = kwargs.get(source)
            if value is not None:
                out[target] = conds.CONDRegular(value)
        return out


class MOSSImageConfig(BASE):
    unet_extra_config = {}
    sampling_settings = {"shift": 1.0, "multiplier": 1000}
    latent_format = latent_formats.Flux2
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    memory_usage_factor = 4.0
    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        return MOSSImage(self, device)

    def clip_target(self, state_dict):
        detect = hunyuan_video.llama_detect(state_dict, "text_encoders.qwen35_9b.transformer.")
        return ClipTarget(MOSSImageTokenizer, te(**detect))


def model_config(state):
    keys = (
        "video_embedder.bias",
        "text_embedder.proj.weight",
        "text_embedder.blocks.0.qkv.weight",
        "shared_block_modulation.weight",
        "layers.0.q_norm.weight",
        "layers.0.proj.ffn_up_gate.weight",
        "final_linear_video.weight",
        "video_head_modulation.weight",
        "time_embedding.0.bias",
    )
    missing = [key for key in keys if key not in state]
    if missing:
        raise ValueError(f"Not a complete moss-image diffusion model; missing: {missing}")
    dim = state["video_embedder.bias"].shape[0]
    head = state["layers.0.q_norm.weight"].shape[0]
    return MOSSImageConfig(
        {
            "dim": dim,
            "num_heads": dim // head,
            "num_layers": model_detection.count_blocks(state, "layers.{}."),
            "ffn_dim": state["layers.0.proj.ffn_up_gate.weight"].shape[0] // 2,
            "latent_channels": state["final_linear_video.weight"].shape[0],
            "text_dim": 4096,
            "freq_dim": 256,
            "rope_base": 10000.0,
        }
    )


def load_model_state(
    state, dtype, disable_dynamic, metadata=None, fp8_optimizations=False, model_options=None
):
    options = {} if model_options is None else model_options
    operations = options.get("custom_operations")
    if operations is None:
        state, metadata = utils.convert_old_quants(state, "", metadata=metadata)
    prefix = model_detection.unet_prefix_from_state_dict(state)
    stripped = utils.state_dict_prefix_replace(state, {prefix: ""}, filter_keys=True)
    if stripped:
        state = stripped
        if operations is None:
            state, metadata = utils.convert_old_quants(state, "", metadata=metadata)
    state = unfuse_block_qkv(state)
    config = model_config(state)
    config.quant_config = utils.detect_layer_quantization(state, "")
    if operations is not None:
        config.custom_operations = operations
    if fp8_optimizations or options.get("fp8_optimizations", False):
        config.optimizations = {**config.optimizations, "fp8": True}
    device = options.get("load_device", model_management.get_torch_device())
    dtype = options.get("dtype", dtype)
    weight_dtype = None if config.quant_config else utils.weight_dtype(state)
    if dtype is None:
        dtype = model_management.unet_dtype(
            model_params=utils.calculate_parameters(state),
            supported_dtypes=config.supported_inference_dtypes,
            weight_dtype=weight_dtype,
        )
    cast = model_management.unet_manual_cast(
        None if config.quant_config else dtype, device, config.supported_inference_dtypes
    )
    config.set_inference_dtype(dtype, cast, device=device)
    model = config.get_model(state)
    patcher_type = model_patcher.ModelPatcher if disable_dynamic else model_patcher.CoreModelPatcher
    offload = options.get("offload_device", model_management.unet_offload_device())
    patcher = patcher_type(
        model, load_device=device, offload_device=offload, fast_disk=storage.state_dict_fast_disk(state)
    )
    if not model_management.is_device_cpu(offload):
        model.to(offload)
    model.load_model_weights(state, "", assign=patcher.is_dynamic())
    return patcher


def load_model(path, weight_dtype, disable_dynamic=False):
    state, metadata = utils.load_torch_file(path, return_metadata=True)
    patcher = load_model_state(
        state,
        DTYPES[weight_dtype],
        disable_dynamic,
        metadata=metadata,
        fp8_optimizations=weight_dtype == "fp8_e4m3fn_fast",
    )
    patcher.cached_patcher_init = (load_model, (path, weight_dtype))
    return patcher


def clip_state(state):
    state = utils.state_dict_prefix_replace(
        state,
        {
            "language_model.": "model.language_model.",
            "visual.": "model.visual.",
        },
    )
    if sd.detect_te_model(state) != sd.TEModel.QWEN35_9B:
        raise ValueError("moss-image requires a Qwen3.5-9B text encoder")
    return utils.state_dict_prefix_replace(
        state,
        {
            "model.language_model.": "model.",
            "model.visual.": "visual.",
        },
    )


def load_clip(path, device, disable_dynamic=False):
    state, metadata = utils.load_torch_file(path, safe_load=True, return_metadata=True)
    state, metadata = utils.convert_old_quants(state, model_prefix="", metadata=metadata)
    clip = load_clip_state(state, device, disable_dynamic, MOSSImageTokenizer, {})
    clip.patcher.cached_patcher_init = (load_clip_patcher, (path, device))
    return clip


def load_clip_state(state, device, disable_dynamic, tokenizer, tokenizer_data):
    state = clip_state(state)
    target = ClipTarget(tokenizer, te(**sd.llama_detect([state])))
    options = {}
    if device == "cpu":
        options = {"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")}
    clip = sd.CLIP(
        target,
        tokenizer_data=tokenizer_data,
        parameters=utils.calculate_parameters(state),
        state_dict=[state],
        model_options=options,
        disable_dynamic=disable_dynamic,
    )
    return clip


def load_clip_patcher(path, device, disable_dynamic=False):
    return load_clip(path, device, disable_dynamic).patcher
