# -*- coding: utf-8 -*-
"""Lightweight data loading utilities."""

import json
import os
from typing import Dict, List
from PIL import Image


def load_questions(path: str) -> List[Dict]:
    """Load a JSONL question file. Each line is a dict with at least
    ``question_id`` and ``image`` fields, optionally ``category``."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_image(image_folder: str, image_name: str) -> Image.Image:
    """Open an image as RGB."""
    path = os.path.join(image_folder, image_name)
    return Image.open(path).convert("RGB")
