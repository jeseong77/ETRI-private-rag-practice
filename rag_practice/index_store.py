from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_practice.chunking import Chunk
from rag_practice.embedding import EmbeddingResult


def save_index(
    path: Path,
    *,
    chunks: list[Chunk],
    embedding_result: EmbeddingResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(chunks) != len(embedding_result.vectors):
        raise ValueError("Chunk 수와 Embedding 벡터 수가 다릅니다.")

    records = []
    for chunk, vector in zip(chunks, embedding_result.vectors):
        records.append({**chunk.to_dict(), "vector": vector})

    payload = {
        "version": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "embedding": {
            "backend": embedding_result.backend,
            "model": embedding_result.model,
            "dimensions": embedding_result.dimensions,
        },
        "chunks": records,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"검색 색인이 없습니다: {path}\n먼저 python ingest.py를 실행하세요."
        )
    return json.loads(path.read_text(encoding="utf-8"))
