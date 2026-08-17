from __future__ import annotations

import argparse

from rag_practice.bootstrap import ensure_embedding_model
from rag_practice.chunking import load_document_chunks
from rag_practice.config import DOCUMENTS_DIR, EMBEDDING_MODEL, INDEX_PATH, OLLAMA_URL
from rag_practice.embedding import EmbeddingUnavailable, create_embeddings
from rag_practice.index_store import save_index


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Markdown 문서를 검색 가능한 RAG 색인으로 변환"
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "ollama", "hash"),
        default="auto",
        help="auto는 Ollama 준비 실패 시 해싱 대체 모드를 사용",
    )
    args = parser.parse_args()

    selected_backend = args.backend
    if args.backend == "auto":
        print("[0/4] 로컬 임베딩 환경을 준비합니다.")
        preparation = ensure_embedding_model(EMBEDDING_MODEL, OLLAMA_URL)
        print(f"      {preparation.message}")
        selected_backend = "ollama" if preparation.ready else "hash"
        if not preparation.ready:
            print("      해싱 대체 모드로 계속 진행합니다.")

    documents, chunks = load_document_chunks(DOCUMENTS_DIR)
    print(f"[1/4] Markdown 문서 {len(documents)}개를 읽었습니다.")
    print(f"[2/4] 문서를 {len(chunks)}개 Chunk로 나눴습니다.")
    if not chunks:
        print("색인할 Chunk가 없습니다.")
        return 1

    try:
        embedding_result = create_embeddings(
            [chunk.text for chunk in chunks],
            requested_backend=selected_backend,
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_URL,
        )
    except EmbeddingUnavailable as error:
        if args.backend != "auto":
            print(f"Embedding 생성 실패: {error}")
            return 1
        print(f"      Ollama Embedding 생성에 실패했습니다: {error}")
        print("      해싱 대체 모드로 다시 시도합니다.")
        embedding_result = create_embeddings(
            [chunk.text for chunk in chunks],
            requested_backend="hash",
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_URL,
        )

    print(
        f"[3/4] {len(embedding_result.vectors)}개 Embedding을 생성했습니다. "
        f"방식={embedding_result.backend}, 차원={embedding_result.dimensions}"
    )
    save_index(INDEX_PATH, chunks=chunks, embedding_result=embedding_result)
    print(f"[4/4] 검색 색인을 저장했습니다: {INDEX_PATH.relative_to(INDEX_PATH.parent.parent)}")
    print("문서 색인 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
