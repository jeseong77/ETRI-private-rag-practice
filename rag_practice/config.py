from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = ROOT_DIR / "documents"
INDEX_PATH = ROOT_DIR / "storage" / "index.json"

EMBEDDING_MODEL = os.environ.get(
    "RAG_EMBEDDING_MODEL",
    "embeddinggemma:300m-qat-q4_0",
)
OLLAMA_URL = os.environ.get("RAG_OLLAMA_URL", "http://127.0.0.1:11434")
HASH_DIMENSIONS = 384
DEFAULT_TOP_K = 3
