from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = ROOT_DIR / "documents"
INDEX_PATH = ROOT_DIR / "storage" / "index.json"

HASH_DIMENSIONS = 384
DEFAULT_TOP_K = 3
