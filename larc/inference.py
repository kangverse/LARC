# -*- coding: utf-8 -*-
"""
LARC inference pipeline.

Given a frozen MLLM and a pre-computed set of Affective Semantic Orientations
(ASO), :func:`larc_inference` performs three stages:

    1. Logit path:    standard multiple-choice classification.
    2. Latent path:   probe an intermediate hidden state with a neutral prompt,
                      project onto the ASO directions, and obtain
                      P_latent(c | x).
    3. RGEM fusion:   compute a Prediction Reliability Index (PRI) from the
                      logit distribution and combine the two paths through a
                      reliability-gated residual, with category relations
                      injected via Relation-Aware Affective Evidence Routing
                      (RAER).

The code supports both the *product* style fusion used during development
(``fusion_mode="product"``) and the equivalent reliability-modulated additive
form described in the paper.
"""

import math
from typing import Any, Dict, Optional

from .vlm.base import BaseVLM
from .aso import AffectiveSemanticOrientations, MultiLayerASO
from .dataset_config import DatasetConfig


# Neutral probe used to read the latent affective state. Re-exported for
# convenience so external scripts only need to import from larc.inference.
NEUTRAL_PROBE_PROMPT = (
    "Please focus on emotion.\n"
    "What emotion does this image express?"
)


# Default 8-way EmoSet prompt, kept for backward-compatible callers that
# do not pass a DatasetConfig.
_CLASSIFY_PROMPT_8CLASS = (
    "Please focus on emotion.\n"
    "Which of the following emotions is represented in the image?\n"
    "A. Amusement\nB. Anger\nC. Awe\nD. Content\n"
    "E. Disgust\nF. Excitement\nG. Fear\nH. Sad\n"
    "Answer with the option's letter from the given choices directly."
)


# ============================================================
# Helpers
# ============================================================

def _compute_entropy(probs: Dict[str, float]) -> float:
    """Shannon entropy of a discrete distribution."""
    h = 0.0
    for p in probs.values():
        if p > 1e-10:
            h -= p * math.log(p)
    return h


def _compute_margin(probs: Dict[str, float]) -> float:
    """Top-1 minus Top-2 probability (lies in [0, 1])."""
    sorted_probs = sorted(probs.values(), reverse=True)
    if len(sorted_probs) < 2:
        return 1.0
    return sorted_probs[0] - sorted_probs[1]


def _compute_pri(probs: Dict[str, float], n_classes: int) -> float:
    """Prediction Reliability Index (paper Eq. (8)):

        PRI = (1 - H(P) / log|C|) * (P_top1 - P_top2)
    """
    if n_classes <= 1:
        return 1.0
    entropy = _compute_entropy(probs)
    max_entropy = math.log(n_classes)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    margin = _compute_margin(probs)
    return (1.0 - norm_entropy) * margin


# ============================================================
# Main entry-point
# ============================================================

def larc_inference(
    vlm: BaseVLM,
    aso: AffectiveSemanticOrientations,
    image: Any,
    # ---- dataset ----
    config: Optional[DatasetConfig] = None,
    # ---- RAER ----
    similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    geo_sigma: float = 0.3,              # RAER temperature sigma_g (Eq. 5)
    eta: float = 0.0,                    # RAER residual smoothing (Eq. 6)
    # ---- latent path ----
    probe_prompt: str = NEUTRAL_PROBE_PROMPT,
    temperature: float = 0.05,           # latent softmax temperature tau
    auto_temperature: bool = True,       # scale tau with number of classes
    # ---- RGEM fusion ----
    fusion_mode: str = "product",        # "product" or "weighted"
    logit_exponent: float = 1.0,         # product mode: P_logit^alpha * P_latent
    topo_weight: float = 0.3,            # weighted mode: topo mixing weight
    # ---- reliability gate ----
    margin_guard: float = 0.15,          # protect logit when its margin is large
    entropy_adaptive: bool = True,       # tune the exponent by P_logit entropy
) -> Dict:
    """One-shot LARC inference for a single image.

    Returns a dict with the following keys (subset shown):

    * ``prediction``         final letter prediction.
    * ``confidence``         final probability of the predicted class.
    * ``logit_prediction``   top-1 letter from the raw logit path.
    * ``logit_top_k``        ``[(letter, prob), ...]`` for the logit path.
    * ``logit_margin``       Top-1 minus Top-2 of the logit distribution.
    * ``logit_entropy``      normalised entropy of the logit distribution.
    * ``topo_prediction``    top-1 letter from the latent (ASO) path.
    * ``topo_top_k``         ``[(letter, prob), ...]`` for the latent path.
    * ``fused_top_k``        ``[(letter, prob), ...]`` after RGEM fusion.
    * ``triggered``          whether the final answer differs from the logit
                             prediction.
    * ``logit_guarded``      True iff the margin guard kept the original logit
                             answer despite a disagreement.
    """
    # Resolve the label set and the multiple-choice prompt.
    if config is not None:
        labels = config.labels
        classify_prompt = config.get_classify_prompt()
    else:
        labels = "ABCDEFGH"
        classify_prompt = _CLASSIFY_PROMPT_8CLASS

    n_classes = len(labels)
    result: Dict = {}

    # ========== Stage 1: logit path ==========
    cls = vlm.classify(image, classify_prompt, labels=labels)
    logit_probs = {l: p for l, p in cls.top_k}

    result["logit_prediction"] = cls.top1_letter
    result["logit_confidence"] = cls.top1_conf
    result["logit_top_k"] = cls.top_k[:5]

    logit_margin = _compute_margin(logit_probs)
    logit_entropy = _compute_entropy(logit_probs)
    max_entropy = math.log(n_classes) if n_classes > 1 else 1.0
    normalized_entropy = logit_entropy / max_entropy if max_entropy > 0 else 0.0
    result["logit_margin"] = logit_margin
    result["logit_entropy"] = normalized_entropy
    result["pri"] = _compute_pri(logit_probs, n_classes)

    # ========== Stage 2: latent (ASO) path ==========
    is_multilayer = isinstance(aso, MultiLayerASO)

    # Slightly soften the latent softmax for finer-grained taxonomies.
    effective_temp = temperature
    if auto_temperature and n_classes != 8:
        effective_temp = temperature * (n_classes / 8.0)
    result["effective_temperature"] = effective_temp

    if is_multilayer:
        hidden = vlm.extract_hidden(image, probe_prompt,
                                    layer_idx=aso.layer_indices[0],
                                    extract_all_layers=True)
        h_dict = {}
        for lid in aso.layer_indices:
            if hidden.all_layers and lid in hidden.all_layers:
                h_dict[lid] = hidden.all_layers[lid]
            else:
                h_dict[lid] = hidden.last_token
        raw_scores = aso.project(h_dict)
        topo_probs = aso.project_softmax(h_dict, temperature=effective_temp)
    else:
        hidden = vlm.extract_hidden(image, probe_prompt, layer_idx=aso.layer_idx)
        raw_scores = aso.project(hidden.last_token)
        topo_probs = aso.project_softmax(hidden.last_token, temperature=effective_temp)

    topo_sorted = sorted(topo_probs.items(), key=lambda x: x[1], reverse=True)
    result["topo_prediction"] = topo_sorted[0][0]
    result["topo_confidence"] = topo_sorted[0][1]
    result["topo_top_k"] = topo_sorted[:5]
    result["raw_scores"] = raw_scores

    # ========== Stage 3: RGEM fusion (with RAER routing) ==========

    # 3a. Entropy-adaptive exponent: low logit-entropy -> trust the logit more.
    if entropy_adaptive:
        if normalized_entropy < 0.5:
            effective_exponent = logit_exponent + (0.5 - normalized_entropy) * 2.0
        else:
            effective_exponent = logit_exponent
    else:
        effective_exponent = logit_exponent
    result["effective_exponent"] = effective_exponent

    # 3b. RAER routing weights w_sigma(c | c_hat_0).
    logit_top1 = cls.top1_letter
    geo_weights = {}
    if similarity_matrix is not None and logit_top1 in similarity_matrix:
        # softmax over the row corresponding to the logit top-1 class.
        sims = []
        for letter in labels:
            sims.append(similarity_matrix[logit_top1].get(letter, 0.0))
        exps = [math.exp(s / max(geo_sigma, 1e-6)) for s in sims]
        z = sum(exps) or 1.0
        for letter, w in zip(labels, exps):
            base = w / z
            # Eq. (6): residual smoothing keeps every class above a floor.
            geo_weights[letter] = (1 - eta) * base + eta / n_classes
    else:
        # No relation graph -> uniform routing (RAER disabled).
        geo_weights = {letter: 1.0 for letter in labels}

    # 3c. Reliability-gated fusion. Two equivalent formulations are kept for
    # convenience; both implement the additive log-residual described in the
    # paper, only the parameterisation differs.
    if fusion_mode == "product":
        fused_raw = {}
        for letter in labels:
            lp = max(logit_probs.get(letter, 0.0), 1e-8)
            tp = max(topo_probs.get(letter, 0.0), 1e-8)
            w = geo_weights.get(letter, 1.0)
            fused_raw[letter] = (lp ** effective_exponent) * tp * w
        total = sum(fused_raw.values())
        if total > 0:
            fused = {k: v / total for k, v in fused_raw.items()}
        else:
            fused = {k: 1.0 / n_classes for k in labels}
    elif fusion_mode == "weighted":
        fused = {}
        for letter in labels:
            lp = logit_probs.get(letter, 0.0)
            tp = topo_probs.get(letter, 0.0)
            fused[letter] = (1 - topo_weight) * lp + topo_weight * tp
    else:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    result["topo_weight"] = topo_weight if fusion_mode == "weighted" else -1.0

    # 3d. Margin guard: protect a clearly-confident logit prediction from being
    # overturned by latent evidence. This is the operational counterpart of
    # PRI gating from the paper.
    fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    fused_top1 = fused_sorted[0][0]
    guarded = False

    if fused_top1 != logit_top1 and logit_margin > margin_guard:
        guarded = True
        result["guard_reason"] = "margin"

    if guarded:
        final_letter = logit_top1
        final_conf = cls.top1_conf
        result["logit_guarded"] = True
    else:
        final_letter = fused_sorted[0][0]
        final_conf = fused_sorted[0][1]
        result["logit_guarded"] = False

    result["prediction"] = final_letter
    result["confidence"] = final_conf
    result["fused_top_k"] = fused_sorted[:5]
    result["triggered"] = (final_letter != logit_top1)

    return result
