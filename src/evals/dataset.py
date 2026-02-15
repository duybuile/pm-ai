"""Golden dataset definitions for Saturn PM assistant evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_PATH = Path(__file__).resolve().parent / "golden_samples.json"


def load_golden_samples() -> list[dict[str, Any]]:
    """Load golden samples from the local JSON dataset file."""
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Golden samples dataset must be a list of test case objects.")

    return payload


GOLDEN_SAMPLES = load_golden_samples()
