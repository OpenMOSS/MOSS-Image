from . import npu_compat

npu_compat.install_npu_cast_guard()

from .nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
