from __future__ import annotations

import argparse

from rag_practice.chunking import load_document_chunks
from rag_practice.config import DOCUMENTS_DIR, EMBEDDING_MODEL, INDEX_PATH, OLLAMA_URL
from rag_practice.embedding import EmbeddingUnavailable, create_embeddings
from rag_practice.index_store import save_index


def main() -> int:
    parser = argparse.ArgumentParser(description="문서를 Chunk와 Embedding 색인으로 변환")
    parser.add_argument(
        "--backend",
        choices=("auto", "ollama", "hash"),
        default="auto",
        help="auto는 Ollama 실패 시 해싱 대체 모드를 사용",
    )
    args = parser.parse_args()

    documents, chunks = load_document_chunks(DOCUMENTS_DIR)
    print(f"[1/4] 문서 {len(documents)}개를 읽었습니다.")
    print(f"[2/4] 문서를 {len(chunks)}개 Chunk로 나눴습니다.")

    if not chunks:
        print("색인할 Chunk가 없습니다.")
        return 1

    try:
        result = create_embeddings(
            [chunk.text for chunk in chunks],
            requested_backend=args.backend,
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_URL,
        )
    except EmbeddingUnavailable as error:
        print(f"임베딩 생성 실패: {error}")
        return 1

    if result.fallback_reason:
        print(f"[경고] {result.fallback_reason}")
        print("[대체 모드] SHA-256 해싱 벡터를 사용합니다.")

    print(
        f"[3/4] {len(result.vectors)}개 Embedding을 생성했습니다. "
        f"방식={result.backend}, 차원={result.dimensions}"
    )
    save_index(INDEX_PATH, chunks=chunks, embedding_result=result)
    print(f"[4/4] 검색 색인을 저장했습니다: {INDEX_PATH.relative_to(INDEX_PATH.parent.parent)}")
    print("색인 생성 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
