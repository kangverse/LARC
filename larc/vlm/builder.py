# -*- coding: utf-8 -*-
"""VLM factory."""

import os

from .base import BaseVLM

# model-path keyword -> canonical model name
_PATH_HINTS = {
    "llava": "llava",
}


def infer_model_name(model_path: str) -> str:
    """Heuristically derive the model family from a path."""
    basename = os.path.basename(model_path.rstrip("/")).lower()
    for hint, name in _PATH_HINTS.items():
        if hint.lower() in basename:
            return name
    raise ValueError(
        f"Cannot infer model type from '{model_path}'. "
        f"Pass --model_name explicitly. Supported: {sorted(set(_PATH_HINTS.values()))}"
    )


def get_model_short_name(model_path: str) -> str:
    """Trailing path component of ``model_path``, used in log/file naming."""
    return os.path.basename(model_path.rstrip("/"))


def build_vlm(model_name: str, model_path: str,
              device: str = "cuda") -> BaseVLM:
    """Construct and load a VLM wrapper."""
    if model_name == "llava":
        from .llava_wrapper import LLaVAWrapper
        vlm = LLaVAWrapper()
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: llava"
        )
    vlm.load(model_path, device)
    return vlm
