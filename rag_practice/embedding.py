from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from rag_practice.config import HASH_DIMENSIONS


TOKEN_PATTERN = re.compile(r"[가-힣]+|[a-z0-9_]+", re.IGNORECASE)


class EmbeddingUnavailable(RuntimeError):
    """요청한 임베딩 방식을 사용할 수 없을 때 발생한다."""


@dataclass(frozen=True)
class EmbeddingResult:
    backend: str
    model: str
    dimensions: int
    vectors: list[list[float]]
    fallback_reason: str | None = None


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


def ollama_embed(
    texts: Sequence[str],
    *,
    model: str,
    base_url: str,
    timeout_seconds: float = 30.0,
) -> list[list[float]]:
    body = json.dumps({"model": model, "input": list(texts)}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise EmbeddingUnavailable(f"Ollama 임베딩 요청 실패: {error}") from error

    vectors = payload.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise EmbeddingUnavailable("Ollama가 입력 수와 다른 임베딩 결과를 반환했습니다.")
    if not vectors or not isinstance(vectors[0], list):
        raise EmbeddingUnavailable("Ollama 임베딩 결과에 벡터가 없습니다.")

    return [[float(value) for value in vector] for vector in vectors]


def create_embeddings(
    texts: Sequence[str],
    *,
    requested_backend: str,
    model: str,
    base_url: str,
) -> EmbeddingResult:
    if requested_backend not in {"auto", "ollama", "hash"}:
        raise ValueError(f"지원하지 않는 임베딩 방식: {requested_backend}")

    if requested_backend == "hash":
        vectors = hash_embed(texts)
        return EmbeddingResult(
            backend="hash",
            model="sha256-hashing-v1",
            dimensions=HASH_DIMENSIONS,
            vectors=vectors,
        )

    try:
        vectors = ollama_embed(texts, model=model, base_url=base_url)
        return EmbeddingResult(
            backend="ollama",
            model=model,
            dimensions=len(vectors[0]),
            vectors=vectors,
        )
    except EmbeddingUnavailable as error:
        if requested_backend == "ollama":
            raise

        vectors = hash_embed(texts)
        return EmbeddingResult(
            backend="hash",
            model="sha256-hashing-v1",
            dimensions=HASH_DIMENSIONS,
            vectors=vectors,
            fallback_reason=str(error),
        )


def embed_query(
    query: str,
    *,
    backend: str,
    model: str,
    base_url: str,
    dimensions: int,
) -> list[float]:
    if backend == "hash":
        return hash_embed([query], dimensions=dimensions)[0]
    if backend == "ollama":
        return ollama_embed([query], model=model, base_url=base_url)[0]
    raise EmbeddingUnavailable(f"색인에 기록된 임베딩 방식을 해석할 수 없습니다: {backend}")

