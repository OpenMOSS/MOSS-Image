"""Ascend NPU compatibility shim for the MOSS-Image ComfyUI plugin.

torch_npu's ``transfer_to_npu`` patches ``nn.Module.to(device)`` to callback
``self.cast_weight(device)``, passing a ``torch.device`` where ComfyUI's
``CastWeightBiasOp.cast_weight(input)`` expects the input tensor. The two
contracts collide in ``comfy.ops.cast_bias_weight``, which reads
``input.dtype`` immediately::

    AttributeError: 'torch.device' object has no attribute 'dtype'

Normalize the argument at that single choke point: if ``input`` is a
``torch.device``, treat it as the target device and leave ``dtype``
untouched. The wrapper is inert for tensor inputs (the stock CUDA flow
never passes a device there), so NVIDIA users see zero behavior change.
"""

import torch

_GUARD_FLAG = "_moss_image_npu_guard"


def install_npu_cast_guard():
    """Wrap ``comfy.ops.cast_bias_weight`` with a device-input normalizer.

    Idempotent; safe to call at plugin import time (custom nodes load after
    core modules import but before any model is loaded, and every core
    call site resolves ``comfy.ops.cast_bias_weight`` dynamically through
    the module namespace).
    """
    import comfy.ops

    current = comfy.ops.cast_bias_weight
    if getattr(current, _GUARD_FLAG, False):
        return

    def cast_bias_weight(s, *args, **kwargs):
        if args:
            first = args[0]
            if isinstance(first, torch.device):
                kwargs.setdefault("device", first)
                args = (None,) + args[1:]
        elif isinstance(kwargs.get("input"), torch.device):
            first = kwargs.pop("input")
            kwargs.setdefault("device", first)
        return current(s, *args, **kwargs)

    setattr(cast_bias_weight, _GUARD_FLAG, True)
    comfy.ops.cast_bias_weight = cast_bias_weight
