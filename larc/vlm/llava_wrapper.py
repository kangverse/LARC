# -*- coding: utf-8 -*-
"""
LLaVA-v1.5 adapter for LARC.

The adapter supports two interfaces:

* :meth:`classify` returns a softmax over the multiple-choice letter options
  using a single ``generate`` step (max_new_tokens=1).
* :meth:`extract_hidden` registers forward hooks on the LLaMA layers and
  returns the last-token (and optionally image-token mean) hidden states at
  the requested layer. The hooks are *read-only*; they do not modify any
  activations.

LARC expects the official LLaVA codebase to be available on the Python path
(its imports begin with ``llava.``). You can do this either by installing
the package (``pip install -e .`` inside the LLaVA repo) or by setting the
``LLAVA_HOME`` environment variable to a directory whose ``llava/`` package
can be imported.
"""

import os
import sys
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .base import BaseVLM, ClassifyResult, HiddenStates


def _ensure_llava_on_path() -> None:
    """Make sure ``import llava`` works.

    Resolution order:

    1. If ``llava`` is already importable, do nothing.
    2. If ``LLAVA_HOME`` is set, prepend it to ``sys.path``.
    3. Otherwise, try a few default sibling locations that match the
       LARC code-release layout.
    """
    try:
        import llava  # noqa: F401
        return
    except ImportError:
        pass

    candidates = []
    env = os.environ.get("LLAVA_HOME")
    if env:
        candidates.append(env)

    here = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        os.path.abspath(os.path.join(here, "..", "..", "third_party", "LLaVA")),
        os.path.abspath(os.path.join(here, "..", "..", "..", "SEPM")),
    ])

    for path in candidates:
        if path and os.path.isdir(os.path.join(path, "llava")):
            sys.path.insert(0, path)
            return

    raise ImportError(
        "Could not locate the `llava` package. Either install it "
        "(`pip install -e .` from the LLaVA repository) or set the "
        "LLAVA_HOME environment variable to a directory whose `llava/` "
        "sub-package is importable."
    )


_ensure_llava_on_path()

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: E402
from llava.conversation import conv_templates                       # noqa: E402
from llava.model.builder import load_pretrained_model               # noqa: E402
from llava.mm_utils import tokenizer_image_token, process_images     # noqa: E402


class LLaVAWrapper(BaseVLM):
    """LLaVA-v1.5 adapter with logit classification and hidden-state probing."""

    # LLaVA-v1.5 uses 336x336 inputs which produce 24*24 = 576 image tokens.
    _NUM_IMAGE_TOKENS = 576

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._label_token_ids = {}
        self._device_val = None
        self._dtype = torch.float16
        self._num_layers = 0
        self._hidden_size = 0
        # Forward-hook state.
        self._hook_outputs = {}

    def load(self, model_path: str, device: str = "cuda") -> None:
        self._device_val = torch.device(device)
        model_name = os.path.basename(model_path.rstrip("/"))
        self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(
            model_path=model_path, model_base=None,
            model_name=model_name, device=device,
        )
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False

        self._num_layers = len(self._model.model.layers)
        self._hidden_size = self._model.config.hidden_size

        # Pre-cache letter token ids for every letter (supports any class count).
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            ids = self._tokenizer.encode(letter, add_special_tokens=False)
            self._label_token_ids[letter] = ids[-1]

    # ==================== internals ====================

    def _prepare_image(self, image: Any) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        return process_images(
            [image], self._image_processor, self._model.config
        )[0].unsqueeze(0).to(device=self._device_val, dtype=self._dtype)

    def _make_input_ids(self, prompt: str) -> torch.Tensor:
        conv = conv_templates["vicuna_v1"].copy()
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        return tokenizer_image_token(
            full_prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
        ).unsqueeze(0).to(self._device_val)

    def _find_image_token_range(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """Locate the IMAGE_TOKEN_INDEX in the prompt. LLaVA replaces it with
        ``_NUM_IMAGE_TOKENS`` image-token embeddings at forward time."""
        ids = input_ids[0].tolist()
        for i, tid in enumerate(ids):
            if tid == IMAGE_TOKEN_INDEX:
                return (i, self._NUM_IMAGE_TOKENS)
        return (0, 0)

    def _make_hook(self, layer_idx: int):
        """Capture the residual stream output of a single decoder layer."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self._hook_outputs[layer_idx] = output[0].detach()
            else:
                self._hook_outputs[layer_idx] = output.detach()
        return hook_fn

    # ==================== logit classification ====================

    def classify(self, image: Any, prompt: str,
                 labels: str = "ABCDEFGH") -> ClassifyResult:
        image_tensor = self._prepare_image(image)
        input_ids = self._make_input_ids(prompt)

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        gen_out = output[-1] if isinstance(output, tuple) and hasattr(output[-1], "scores") \
            else (output[0] if isinstance(output, tuple) else output)

        scores = gen_out.scores[0][0]
        label_ids = [self._label_token_ids[l] for l in labels]
        label_logits = scores[label_ids]
        label_probs = F.softmax(label_logits.float(), dim=0)

        top_k = sorted(
            [(letter, label_probs[i].item()) for i, letter in enumerate(labels)],
            key=lambda x: x[1], reverse=True,
        )

        return ClassifyResult(
            top_k=top_k,
            top1_letter=top_k[0][0],
            top1_conf=top_k[0][1],
        )

    # ==================== hidden-state probing ====================

    def extract_hidden(
        self,
        image: Any,
        prompt: str,
        layer_idx: int = -1,
        extract_all_layers: bool = False,
    ) -> HiddenStates:
        """Extract intermediate hidden states via forward hooks.

        We reuse the same ``generate`` call as :meth:`classify` so the
        forward path is identical, and rely on hooks to read the residual
        stream after each target layer. Hooks are always removed in a
        ``finally`` block so subsequent calls remain clean.
        """
        if layer_idx < 0:
            layer_idx = int(self._num_layers * 0.65)

        image_tensor = self._prepare_image(image)
        input_ids = self._make_input_ids(prompt)
        img_start, num_img_tokens = self._find_image_token_range(input_ids)

        self._hook_outputs.clear()
        handles = []

        target_layers = list(range(self._num_layers)) if extract_all_layers \
            else [layer_idx]
        for lid in target_layers:
            layer = self._model.model.layers[lid]
            handles.append(layer.register_forward_hook(self._make_hook(lid)))

        try:
            with torch.no_grad():
                self._model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
        finally:
            for h in handles:
                h.remove()

        target_output = self._hook_outputs.get(layer_idx)
        if target_output is None:
            raise RuntimeError(
                f"Failed to capture hidden states at layer {layer_idx}"
            )

        seq_len = target_output.shape[1]
        last_token_h = target_output[0, -1, :].float().cpu()

        image_mean_h = None
        if num_img_tokens > 0 and img_start + num_img_tokens <= seq_len:
            img_h = target_output[0, img_start:img_start + num_img_tokens, :]
            image_mean_h = img_h.float().mean(dim=0).cpu()

        all_layers_dict = None
        if extract_all_layers:
            all_layers_dict = {
                lid: h[0, -1, :].float().cpu()
                for lid, h in self._hook_outputs.items()
            }

        self._hook_outputs.clear()

        return HiddenStates(
            last_token=last_token_h,
            image_mean=image_mean_h,
            all_layers=all_layers_dict,
            layer_idx=layer_idx,
            num_image_tokens=num_img_tokens,
        )

    # ==================== properties ====================

    def get_num_layers(self) -> int:
        return self._num_layers

    def get_hidden_size(self) -> int:
        return self._hidden_size

    @property
    def device(self) -> torch.device:
        return self._device_val
