"""Central configuration: thresholds, limits and paths (overridable with environment variables)."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PACKAGE_DIR.parent
REPO_ROOT = APP_DIR.parent.parent
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", APP_DIR / "artefactos"))
DATA_DIR = REPO_ROOT / "avance2" / "datos"

VERSION = "1.0.0"

# Decision thresholds (same values as the Avance 2 policy)
TEXT_THRESHOLD = 0.40       # below this the intent classifier is not trusted
AMBIGUOUS_TEXT = 0.60       # with a reliable photo, a text below this lets the image decide
IMAGE_THRESHOLD = 0.60      # below this the photo is treated as "otro"
TONE_THRESHOLD = 0.60       # confidence needed to act on the trained tone model

# Input limits
MAX_TEXT_CHARS = 500
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MIN_IMAGE_SIDE = 32
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FALLBACKS = 2           # consecutive misunderstood messages before escalating to the host


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


API_TOKEN = os.environ.get("API_TOKEN", "")                 # empty = no token required
RATE_LIMIT_PER_MIN = env_int("RATE_LIMIT_PER_MIN", 60)      # per client; 0 disables the limit
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
VISION_BACKBONE = os.environ.get("VISION_BACKBONE", "")     # empty = the one chosen in vision_meta.json
