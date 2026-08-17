from __future__ import annotations

import argparse

from rag_practice.config import DEFAULT_TOP_K, INDEX_PATH, OLLAMA_URL
from rag_practice.embedding import EmbeddingUnavailable
from rag_practice.index_store import load_index
from rag_practice.retrieval import search_index


def main() -> int:
    parser = argparse.ArgumentParser(description="질문과 관련된 내부 규정 검색")
    parser.add_argument("question", help="검색할 자연어 질문")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    index = load_index(INDEX_PATH)
    embedding = index["embedding"]
    print(f"색인 방식: {embedding['backend']} / {embedding['model']}")
    print(f"질문: {args.question}")
    print()

    try:
        results = search_index(
            args.question,
            index=index,
            base_url=OLLAMA_URL,
            top_k=args.top_k,
        )
    except EmbeddingUnavailable as error:
        print(f"검색 실패: {error}")
        print("현재 환경에 맞게 python ingest.py를 다시 실행하세요.")
        return 1

    for number, result in enumerate(results, start=1):
        print(f"{number}위 · 유사도 {result.score:.4f}")
        print(f"출처: {result.source} · {result.heading}")
        print(result.text)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

