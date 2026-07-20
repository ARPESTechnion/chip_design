"""
Top-level configuration for the full pipeline (see all_code.py): flake_detection
(rank/select candidate images) -> grid_preperation (export selected images to CIF
and place them into the master template).

Holds no tunable parameters of its own -- those live in each stage's own config
(DetectionConfig, CifExportConfig). This just points at which stage config files
to use, so there's one file to edit/version instead of duplicating paths across
callers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    detection_config_path: str = "flake_detection/config/orange_flakes_tuned.json"
    """DetectionConfig JSON for the flake_detection stage (image scoring/ranking)."""

    cif_export_config_path: Optional[str] = "grid_preperation/config/example_config.json"
    """CifExportConfig JSON for the grid_preperation stage (CIF export/placement).
    If None, CifExportConfig()'s built-in defaults are used.

    Note: whatever detection_config_path is set above always takes precedence over
    this file's own 'detection_config_path' field once loaded -- see all_code.py --
    so the two stages never score/mask against different color or marker ranges.
    """

    def save(self, path: str | Path) -> None:
        """Persist config to JSON for reproducibility."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        """Load config from a JSON file. Keys prefixed with '_' are comments."""
        with open(path) as f:
            data = json.load(f)
        data = {k: v for k, v in data.items() if not k.startswith('_')}
        return cls(**data)
