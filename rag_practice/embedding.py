from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from rag_practice.config import HASH_DIMENSIONS


TOKEN_PATTERN = re.compile(r"[가-힣]+|[a-z0-9_]+", re.IGNORECASE)


@dataclass(frozen=True)
class EmbeddingResult:
    backend: str
    model: str
    dimensions: int
    vectors: list[list[float]]


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _hash_features(text: str) -> list[str]:
    words = [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
    features = list(words)
    features.extend(f"{left}::{right}" for left, right in zip(words, words[1:]))

    korean_text = "".join(re.findall(r"[가-힣]", text))
    features.extend(
        korean_text[index : index + 3]
        for index in range(max(0, len(korean_text) - 2))
    )
    return features


def hash_embed(texts: Sequence[str], dimensions: int = HASH_DIMENSIONS) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimensions
        for feature, count in Counter(_hash_features(text)).items():
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            position = int.from_bytes(digest[:4], "big") % dimensions
            direction = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[position] += direction * (1.0 + math.log(count))
        vectors.append(_normalize(vector))
    return vectors


def create_hash_embeddings(texts: Sequence[str]) -> EmbeddingResult:
    vectors = hash_embed(texts)
    return EmbeddingResult(
        backend="hash",
        model="sha256-hashing-v1",
        dimensions=HASH_DIMENSIONS,
        vectors=vectors,
    )


def embed_query(query: str, dimensions: int) -> list[float]:
    return hash_embed([query], dimensions=dimensions)[0]
