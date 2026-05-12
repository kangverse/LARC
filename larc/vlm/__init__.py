# -*- coding: utf-8 -*-
"""Vision-language model wrappers used by LARC.

LARC operates at the representation level, so each VLM wrapper must support
both standard logit-based classification *and* extraction of intermediate
hidden states via forward hooks. See :mod:`larc.vlm.base` for the API.
"""

from .base import BaseVLM, ClassifyResult, HiddenStates
from .builder import build_vlm, infer_model_name, get_model_short_name

__all__ = [
    "BaseVLM", "ClassifyResult", "HiddenStates",
    "build_vlm", "infer_model_name", "get_model_short_name",
]
