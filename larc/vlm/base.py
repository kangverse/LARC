# -*- coding: utf-8 -*-
"""
VLM abstract interface.

A LARC-compatible VLM must expose both classification (logit path) and
hidden-state extraction (latent path). The latter is implemented via forward
hooks on the language-model layers, and returns the last-token hidden state
plus, optionally, the mean of the image tokens and the hidden states at all
layers. All tensors are returned on CPU to keep memory bounded.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class ClassifyResult:
    """Output of :meth:`BaseVLM.classify`."""
    top_k: List[Tuple[str, float]]  # [(letter, prob), ...] sorted by prob desc
    top1_letter: str
    top1_conf: float


@dataclass
class HiddenStates:
    """Output of :meth:`BaseVLM.extract_hidden`."""
    # Hidden state of the last token at the target layer: [hidden_dim].
    last_token: torch.Tensor
    # Mean of the image-token hidden states at the target layer: [hidden_dim].
    image_mean: Optional[torch.Tensor] = None
    # All layers' last-token hidden states keyed by layer index.
    all_layers: Optional[Dict[int, torch.Tensor]] = None
    # Metadata.
    layer_idx: int = -1
    num_image_tokens: int = 0


class BaseVLM(ABC):
    """Abstract base class for VLM adapters."""

    @abstractmethod
    def load(self, model_path: str, device: str = "cuda") -> None: ...

    @abstractmethod
    def classify(self, image: Any, prompt: str,
                 labels: str = "ABCDEFGH") -> ClassifyResult:
        """Logit-based classification. ``labels`` is the set of valid letter
        options, e.g. ``"ABCDEF"`` for 6 classes."""
        ...

    @abstractmethod
    def extract_hidden(
        self,
        image: Any,
        prompt: str,
        layer_idx: int = -1,
        extract_all_layers: bool = False,
    ) -> HiddenStates:
        """Extract intermediate hidden states.

        Args:
            image: PIL Image or path.
            prompt: text prompt.
            layer_idx: target layer index. ``-1`` selects ~65% of depth.
            extract_all_layers: also return last-token hidden states for every
                layer (useful for layer-selection ablation).
        """
        ...

    @abstractmethod
    def get_num_layers(self) -> int: ...

    @abstractmethod
    def get_hidden_size(self) -> int: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...
