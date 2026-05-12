# -*- coding: utf-8 -*-
"""
Affective Semantic Orientation (ASO).

For each emotion category ``c``, we estimate a discriminative direction
:math:`\mathbf{a}_c` in the latent space of a frozen MLLM. The direction
serves as a class-level anchor against which test-sample hidden states are
projected to obtain a *latent evidence distribution* (used by LARC's
inference pipeline together with RAER and RGEM).

This module provides several variants for constructing the ASO directions:

* :func:`extract_aso` (default): class-centroid contrasted with the global
  mean of the calibration set, then L2-normalised. Simple, training-free,
  works well in practice with very small calibration sets.
* :func:`extract_aso_pca`: PCA on the positive/negative prompt deltas; uses
  the first principal component as the direction. Slightly more robust when
  the calibration set is tiny.
* :func:`extract_aso_multilayer`: builds ASO at several intermediate layers
  and aggregates the projection scores at inference time.

The output object (:class:`AffectiveSemanticOrientations` or
:class:`MultiLayerASO`) carries everything the inference pipeline needs:
the per-category direction(s), the target layer index, hidden-size, and
the category-vs-category cosine-similarity matrix (used by RAER).
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .vlm.base import BaseVLM


# Letter-to-emotion fallback (used only for log lines when the caller does
# not pass an explicit mapping).
_DEFAULT_LETTER2EMO: Dict[str, str] = {
    "A": "amusement", "B": "anger", "C": "awe", "D": "content",
    "E": "disgust", "F": "excitement", "G": "fear", "H": "sad",
}


# ============================================================
# Prompt templates for contrastive ASO extraction
# ============================================================

def _make_contrastive_prompts(emotion_name: str) -> Tuple[str, str]:
    """Build a (positive, negative) prompt pair that differ only in the
    emotion label, so that the hidden-state difference isolates the
    affective concept."""
    positive = (
        f"Please focus on emotion.\n"
        f"The dominant emotion expressed in this image is {emotion_name}."
    )
    negative = (
        f"Please focus on emotion.\n"
        f"The dominant emotion expressed in this image is not {emotion_name}."
    )
    return positive, negative


# ============================================================
# Container objects
# ============================================================

class AffectiveSemanticOrientations:
    """A set of category-level ASO directions extracted at a single layer."""

    def __init__(self):
        # {letter: tensor[hidden_dim]}
        self.vectors: Dict[str, torch.Tensor] = {}
        self.layer_idx: int = -1
        self.hidden_size: int = 0
        self.num_calibration_images: int = 0
        self.metadata: Dict = {}

    # -------------------- inference-side projection --------------------

    def project(self, hidden_state: torch.Tensor) -> Dict[str, float]:
        """Project a hidden state onto every ASO direction.

        Returns a dict ``{letter: cosine_similarity}``."""
        h = hidden_state.float()
        scores = {}
        for letter, vec in self.vectors.items():
            scores[letter] = F.cosine_similarity(
                h.unsqueeze(0), vec.unsqueeze(0)
            ).item()
        return scores

    def project_softmax(self, hidden_state: torch.Tensor,
                        temperature: float = 1.0) -> Dict[str, float]:
        """Project then convert into a softmax distribution over categories."""
        raw_scores = self.project(hidden_state)
        letters = sorted(raw_scores.keys())
        logits = torch.tensor([raw_scores[l] for l in letters]) / temperature
        probs = F.softmax(logits, dim=0)
        return {l: probs[i].item() for i, l in enumerate(letters)}

    # -------------------- RAER support --------------------

    def similarity_matrix(self) -> Dict[str, Dict[str, float]]:
        """Pairwise cosine similarity between ASO directions. RAER uses this
        as the affective relation graph G."""
        letters = sorted(self.vectors.keys())
        S: Dict[str, Dict[str, float]] = {}
        for li in letters:
            S[li] = {}
            for lj in letters:
                S[li][lj] = F.cosine_similarity(
                    self.vectors[li].unsqueeze(0),
                    self.vectors[lj].unsqueeze(0),
                ).item()
        return S

    def print_similarity_matrix(self,
                                letter2emo: Optional[Dict[str, str]] = None) -> None:
        """Print the relation matrix as a small table for sanity-checking."""
        _l2e = letter2emo if letter2emo else _DEFAULT_LETTER2EMO
        letters = sorted(self.vectors.keys())
        S = self.similarity_matrix()
        print(f"\n{'':>4}", end="")
        for l in letters:
            emo = _l2e.get(l, l)[:6]
            print(f" {emo:>7}", end="")
        print()
        for li in letters:
            emo_i = _l2e.get(li, li)[:6]
            print(f"{emo_i:>4}", end="")
            for lj in letters:
                print(f" {S[li][lj]:>7.3f}", end="")
            print()

    # -------------------- IO --------------------

    def save(self, path: str) -> None:
        data = {
            "vectors": {k: v.numpy().tolist() for k, v in self.vectors.items()},
            "layer_idx": self.layer_idx,
            "hidden_size": self.hidden_size,
            "num_calibration_images": self.num_calibration_images,
            "metadata": self.metadata,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"ASO directions saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AffectiveSemanticOrientations":
        with open(path, "r") as f:
            data = json.load(f)
        aso = cls()
        aso.vectors = {k: torch.tensor(v) for k, v in data["vectors"].items()}
        aso.layer_idx = data["layer_idx"]
        aso.hidden_size = data["hidden_size"]
        aso.num_calibration_images = data.get("num_calibration_images", 0)
        aso.metadata = data.get("metadata", {})
        return aso


class MultiLayerASO:
    """An ensemble of single-layer ASO directions extracted at several layers."""

    def __init__(self):
        # {layer_idx: AffectiveSemanticOrientations}
        self.layer_directions: Dict[int, AffectiveSemanticOrientations] = {}
        self.layer_indices: List[int] = []
        self.hidden_size: int = 0
        self.num_calibration_images: int = 0

    @property
    def layer_idx(self) -> int:
        """Return the median layer index for compatibility with single-layer API."""
        if self.layer_indices:
            return self.layer_indices[len(self.layer_indices) // 2]
        return -1

    def project(self,
                hidden_states_dict: Dict[int, torch.Tensor]) -> Dict[str, float]:
        """Average projection scores across the configured layers."""
        avg_scores: Dict[str, float] = {}
        n_layers = 0
        for lid, aso in self.layer_directions.items():
            if lid in hidden_states_dict:
                scores = aso.project(hidden_states_dict[lid])
                for letter, score in scores.items():
                    avg_scores[letter] = avg_scores.get(letter, 0.0) + score
                n_layers += 1
        if n_layers > 0:
            avg_scores = {k: v / n_layers for k, v in avg_scores.items()}
        return avg_scores

    def project_softmax(self, hidden_states_dict: Dict[int, torch.Tensor],
                        temperature: float = 1.0) -> Dict[str, float]:
        raw_scores = self.project(hidden_states_dict)
        letters = sorted(raw_scores.keys())
        logits = torch.tensor([raw_scores[l] for l in letters]) / temperature
        probs = F.softmax(logits, dim=0)
        return {l: probs[i].item() for i, l in enumerate(letters)}

    def similarity_matrix(self) -> Dict[str, Dict[str, float]]:
        """Use the median layer's similarity matrix for RAER routing."""
        mid_idx = self.layer_indices[len(self.layer_indices) // 2]
        return self.layer_directions[mid_idx].similarity_matrix()

    def print_similarity_matrix(self,
                                letter2emo: Optional[Dict[str, str]] = None) -> None:
        mid_idx = self.layer_indices[len(self.layer_indices) // 2]
        print(f"(Showing layer {mid_idx} similarity matrix)")
        self.layer_directions[mid_idx].print_similarity_matrix(letter2emo)

    def save(self, path: str) -> None:
        data = {
            "layer_indices": self.layer_indices,
            "hidden_size": self.hidden_size,
            "num_calibration_images": self.num_calibration_images,
            "layers": {},
        }
        for lid, aso in self.layer_directions.items():
            data["layers"][str(lid)] = {k: v.numpy().tolist()
                                        for k, v in aso.vectors.items()}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Multi-layer ASO directions saved to {path}")


# ============================================================
# Extractors
# ============================================================

NEUTRAL_PROBE_PROMPT = (
    "Please focus on emotion.\n"
    "What emotion does this image express?"
)


def extract_aso(
    vlm: BaseVLM,
    class_images: Dict[str, List[Any]],
    layer_idx: int = -1,
    probe_prompt: str = NEUTRAL_PROBE_PROMPT,
    letter2emo: Optional[Dict[str, str]] = None,
) -> AffectiveSemanticOrientations:
    """Centroid-based ASO extraction (default).

    For each calibration image we read the hidden state at ``layer_idx`` using
    a *class-agnostic* probe prompt. The direction for category ``c`` is the
    L2-normalised deviation between its class centroid and the global mean
    of all calibration samples, i.e. a contrast against the rest of the
    classes seen so far.

    Args:
        vlm: a loaded :class:`BaseVLM`.
        class_images: ``{letter: [PIL.Image, ...]}`` calibration images.
        layer_idx: target Transformer layer index, ``-1`` selects ~65% depth.
        probe_prompt: class-agnostic probe; default matches the paper.
        letter2emo: optional letter -> human-readable name for logging.

    Returns:
        :class:`AffectiveSemanticOrientations`.
    """
    if layer_idx < 0:
        layer_idx = int(vlm.get_num_layers() * 0.65)

    total_imgs = sum(len(v) for v in class_images.values())
    print(f"Extracting ASO directions at layer {layer_idx}/{vlm.get_num_layers()}")
    print(f"Using {total_imgs} calibration images across {len(class_images)} classes")

    aso = AffectiveSemanticOrientations()
    aso.layer_idx = layer_idx
    aso.hidden_size = vlm.get_hidden_size()
    aso.num_calibration_images = total_imgs
    aso.metadata["method"] = "centroid"

    _l2e = letter2emo if letter2emo else _DEFAULT_LETTER2EMO

    # Pass 1: gather per-class hidden-state means.
    centroids: Dict[str, torch.Tensor] = {}
    all_hiddens: List[torch.Tensor] = []

    for letter in sorted(class_images.keys()):
        emotion_name = _l2e.get(letter, letter)
        images = class_images[letter]
        hiddens = []
        for img in images:
            h = vlm.extract_hidden(img, probe_prompt, layer_idx=layer_idx)
            hiddens.append(h.last_token)
            all_hiddens.append(h.last_token)
        centroid = torch.stack(hiddens).mean(dim=0)
        centroids[letter] = centroid
        print(f"  [{letter}] {emotion_name:>12}: n={len(images)}, "
              f"centroid_norm={centroid.norm():.4f}")

    # Pass 2: each ASO direction = (class centroid - global mean), L2-normalised.
    global_mean = torch.stack(all_hiddens).mean(dim=0)
    print(f"  Global mean norm: {global_mean.norm():.4f}")

    for letter, centroid in centroids.items():
        direction = centroid - global_mean
        aso.vectors[letter] = F.normalize(direction, dim=0)
        print(f"  [{letter}] direction norm={direction.norm():.4f}")

    return aso


def extract_aso_pca(
    vlm: BaseVLM,
    calibration_images: List[Any],
    layer_idx: int = -1,
    letter2emo: Optional[Dict[str, str]] = None,
) -> AffectiveSemanticOrientations:
    """PCA variant: for each category we collect positive/negative prompt
    deltas across the calibration set and use the first principal component
    as the ASO direction. Useful when the calibration set is tiny.
    """
    if layer_idx < 0:
        layer_idx = int(vlm.get_num_layers() * 0.65)

    print(f"Extracting ASO directions (PCA) at layer "
          f"{layer_idx}/{vlm.get_num_layers()}")

    aso = AffectiveSemanticOrientations()
    aso.layer_idx = layer_idx
    aso.hidden_size = vlm.get_hidden_size()
    aso.num_calibration_images = len(calibration_images)
    aso.metadata["method"] = "pca"

    _l2e = letter2emo if letter2emo else _DEFAULT_LETTER2EMO

    for letter in sorted(_l2e.keys()):
        emotion_name = _l2e[letter]
        pos_prompt, neg_prompt = _make_contrastive_prompts(emotion_name)

        deltas = []
        for img in calibration_images:
            h_pos = vlm.extract_hidden(img, pos_prompt, layer_idx=layer_idx)
            h_neg = vlm.extract_hidden(img, neg_prompt, layer_idx=layer_idx)
            deltas.append(h_pos.last_token - h_neg.last_token)
        delta_matrix = torch.stack(deltas)
        delta_centered = delta_matrix - delta_matrix.mean(dim=0)

        if delta_centered.shape[0] >= 2:
            U, S, Vh = torch.linalg.svd(delta_centered, full_matrices=False)
            principal = Vh[0]
            mean_delta = delta_matrix.mean(dim=0)
            if torch.dot(principal, mean_delta) < 0:
                principal = -principal
            direction = F.normalize(principal, dim=0)
            var_explained = (S[0] ** 2 / (S ** 2).sum()).item()
        else:
            direction = F.normalize(delta_matrix.mean(dim=0), dim=0)
            var_explained = 1.0

        aso.vectors[letter] = direction
        print(f"  [{letter}] {emotion_name:>12}: var_explained={var_explained:.3f}")

    return aso


def extract_aso_multilayer(
    vlm: BaseVLM,
    class_images: Dict[str, List[Any]],
    layer_indices: Optional[List[int]] = None,
    probe_prompt: str = NEUTRAL_PROBE_PROMPT,
    letter2emo: Optional[Dict[str, str]] = None,
) -> MultiLayerASO:
    """Build a multi-layer ASO ensemble.

    By default, we sample five evenly-spaced intermediate layers from 50%
    to 80% of model depth and average the projection scores at inference.
    """
    num_layers = vlm.get_num_layers()
    if layer_indices is None:
        layer_indices = sorted(
            {int(num_layers * r) for r in [0.50, 0.575, 0.65, 0.725, 0.80]}
        )

    total_imgs = sum(len(v) for v in class_images.values())
    print("Extracting multi-layer ASO directions")
    print(f"  Layers: {layer_indices}")
    print(f"  Images: {total_imgs} across {len(class_images)} classes")

    mlaso = MultiLayerASO()
    mlaso.layer_indices = layer_indices
    mlaso.hidden_size = vlm.get_hidden_size()
    mlaso.num_calibration_images = total_imgs

    # Pass 1: hidden states per (layer, class).
    layer_hiddens: Dict[int, Dict[str, List[torch.Tensor]]] = {
        lid: {l: [] for l in class_images.keys()} for lid in layer_indices
    }
    all_hiddens: Dict[int, List[torch.Tensor]] = {lid: [] for lid in layer_indices}
    _l2e = letter2emo if letter2emo else _DEFAULT_LETTER2EMO

    for letter in sorted(class_images.keys()):
        emotion_name = _l2e.get(letter, letter)
        images = class_images[letter]
        print(f"  [{letter}] {emotion_name:>12}: extracting {len(images)} images...",
              end="", flush=True)
        for img in images:
            h = vlm.extract_hidden(img, probe_prompt,
                                   layer_idx=layer_indices[0],
                                   extract_all_layers=True)
            for lid in layer_indices:
                if h.all_layers and lid in h.all_layers:
                    vec = h.all_layers[lid]
                else:
                    vec = h.last_token
                layer_hiddens[lid][letter].append(vec)
                all_hiddens[lid].append(vec)
        print(" done")

    # Pass 2: build a single-layer ASO per layer.
    for lid in layer_indices:
        aso = AffectiveSemanticOrientations()
        aso.layer_idx = lid
        aso.hidden_size = vlm.get_hidden_size()
        aso.num_calibration_images = total_imgs
        aso.metadata["method"] = "centroid_multilayer"
        global_mean = torch.stack(all_hiddens[lid]).mean(dim=0)
        for letter in sorted(class_images.keys()):
            hiddens = layer_hiddens[lid][letter]
            centroid = torch.stack(hiddens).mean(dim=0)
            direction = centroid - global_mean
            aso.vectors[letter] = F.normalize(direction, dim=0)
        mlaso.layer_directions[lid] = aso
        print(f"  Layer {lid}: ASO directions computed")

    return mlaso
