# -*- coding: utf-8 -*-
"""
Dataset configuration registry.

Each dataset has a unique number of emotion categories and an associated
multiple-choice prompt. The :class:`DatasetConfig` dataclass handles letter
indexing, prompt construction, and (optionally) Russell's circumplex
coordinates for affective analysis.

Supported datasets:
    * emoset    EmoSet-118K  8-way
    * emotion6  Emotion6     6-way
    * abstract  Abstract     8-way
    * artphoto  ArtPhoto     8-way
    * webemo7   WEBEmo       7-way (coarse)
    * webemo25  WEBEmo       25-way (fine-grained)

Default image / question file paths point to a shared dataset directory and
can be overridden at run time via CLI flags.
"""

import os
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class DatasetConfig:
    """A dataset configuration."""
    name: str
    classes: List[str]                                                    # ordered category names
    image_folder: str = ""                                                # image root directory
    question_file: str = ""                                               # question JSONL path
    circumplex_coords: Optional[Dict[str, Tuple[float, float]]] = None    # letter -> (valence, arousal)

    def __post_init__(self):
        n = len(self.classes)
        if n > 26:
            raise ValueError(f"At most 26 classes (A-Z) supported, got {n}")
        letters = list(string.ascii_uppercase[:n])
        self._labels = "".join(letters)
        self._letter2emo = {l: c for l, c in zip(letters, self.classes)}
        self._emo2letter = {c: l for l, c in zip(letters, self.classes)}

    # ---------- properties ----------

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def labels(self) -> str:
        """The available letter labels as a single string, e.g. ``'ABCDEFGH'``."""
        return self._labels

    @property
    def letter2emo(self) -> Dict[str, str]:
        return dict(self._letter2emo)

    @property
    def emo2letter(self) -> Dict[str, str]:
        return dict(self._emo2letter)

    # ---------- prompt generation ----------

    def get_classify_prompt(self) -> str:
        """Construct the VLM multiple-choice prompt from the class list."""
        lines = ["Please focus on emotion.",
                 "Which of the following emotions is represented in the image?"]
        for letter, emo in self._letter2emo.items():
            display = emo.replace("_", " ").title()
            lines.append(f"{letter}. {display}")
        lines.append("Answer with the option's letter from the given choices directly.")
        return "\n".join(lines)

    # ---------- circumplex helpers ----------

    def get_valence(self, letter: str) -> str:
        if self.circumplex_coords and letter in self.circumplex_coords:
            return "positive" if self.circumplex_coords[letter][0] > 0 else "negative"
        return "unknown"

    def get_arousal(self, letter: str) -> str:
        if self.circumplex_coords and letter in self.circumplex_coords:
            return "high" if self.circumplex_coords[letter][1] > 0 else "low"
        return "unknown"

    def has_circumplex(self) -> bool:
        return self.circumplex_coords is not None and len(self.circumplex_coords) > 0


# ============================================================
# Default dataset registry
# ------------------------------------------------------------
# All paths are relative to the project root by default. To point them at a
# different location, either set the ``LARC_DATA_ROOT`` environment variable
# or override ``--image_folder`` / ``--question_file`` on the command line.
# ============================================================

_BASE_DATA = os.environ.get("LARC_DATA_ROOT", "data")

PRESETS: Dict[str, DatasetConfig] = {}


def _register(name: str, config: DatasetConfig):
    PRESETS[name] = config



# WEBEmo 25 -> 7 mapping (used to derive 7-way labels from 25-way folders).
WEBEMO_25_TO_7 = {
    "affection": "love", "cheerfullness": "joy", "confusion": "confusion",
    "contentment": "joy", "disappointment": "sadness", "disgust": "anger",
    "enthrallment": "joy", "envy": "anger", "exasperation": "anger",
    "gratitude": "love", "horror": "fear", "irritabilty": "anger",
    "lust": "love", "neglect": "sadness", "nervousness": "fear",
    "optimism": "joy", "pride": "joy", "rage": "anger",
    "relief": "joy", "sadness": "sadness", "shame": "sadness",
    "suffering": "sadness", "surprise": "surprise", "sympathy": "sadness",
    "zest": "joy",
}


def get_config(name: str) -> DatasetConfig:
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return PRESETS[name]


def list_datasets() -> List[str]:
    return sorted(PRESETS.keys())
