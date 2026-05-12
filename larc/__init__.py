# -*- coding: utf-8 -*-
"""
LARC: Latent Affective Representation Calibration.

A training-free inference framework that reads latent affective evidence from
the intermediate layers of a frozen MLLM and calibrates the final logit-based
prediction. LARC consists of three components:

    * ASO   - Affective Semantic Orientation
    * RAER  - Relation-Aware Affective Evidence Routing
    * RGEM  - Reliability-Gated Evidence Modulation

The public entry-point is :func:`larc.inference.larc_inference`.
"""

from .aso import (
    AffectiveSemanticOrientations,
    MultiLayerASO,
    extract_aso,
    extract_aso_pca,
    extract_aso_multilayer,
)
from .inference import larc_inference, NEUTRAL_PROBE_PROMPT
from .dataset_config import DatasetConfig, get_config, list_datasets

__all__ = [
    "AffectiveSemanticOrientations",
    "MultiLayerASO",
    "extract_aso",
    "extract_aso_pca",
    "extract_aso_multilayer",
    "larc_inference",
    "NEUTRAL_PROBE_PROMPT",
    "DatasetConfig",
    "get_config",
    "list_datasets",
]
