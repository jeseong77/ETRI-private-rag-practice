from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_practice.embedding import embed_query


@dataclass(frozen=True)
class SearchResult:
    score: float
    source: str
    heading: str
    text: str


def _dot_product(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("질문 벡터와 문서 벡터의 차원이 다릅니다.")
    return sum(a * b for a, b in zip(left, right))


def search_index(
    query: str,
    *,
    index: dict[str, Any],
    base_url: str,
    top_k: int,
) -> list[SearchResult]:
    embedding = index["embedding"]
    query_vector = embed_query(
        query,
        backend=embedding["backend"],
        model=embedding["model"],
        base_url=base_url,
        dimensions=int(embedding["dimensions"]),
    )

    ranked = sorted(
        index["chunks"],
        key=lambda chunk: _dot_product(query_vector, chunk["vector"]),
        reverse=True,
    )
    return [
        SearchResult(
            score=_dot_product(query_vector, chunk["vector"]),
            source=chunk["source"],
            heading=chunk["heading"],
            text=chunk["text"],
        )
        for chunk in ranked[:top_k]
    ]
