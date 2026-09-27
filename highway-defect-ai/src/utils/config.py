"""Small YAML config loader shared across the pipeline."""
from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def project_root() -> Path:
    """Returns the repo root, assuming this file lives at src/utils/config.py."""
    return Path(__file__).resolve().parents[2]
