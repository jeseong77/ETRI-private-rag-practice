from __future__ import annotations

import argparse

from rag_practice.bootstrap import ensure_embedding_model
from rag_practice.chunking import load_document_chunks
from rag_practice.config import (
    DEFAULT_TOP_K,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    INDEX_PATH,
    OLLAMA_URL,
)
from rag_practice.embedding import EmbeddingUnavailable, create_embeddings
from rag_practice.generation import build_rag_prompt, generate_with_ollama
from rag_practice.index_store import save_index
from rag_practice.retrieval import search_index


DEFAULT_QUESTION = "USB를 외부로 반출하려면 어떤 승인이 필요한가?"


def main() -> int:
    parser = argparse.ArgumentParser(description="Private RAG 전체 흐름 한 번에 실행")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--chat-model", help="설치된 Ollama 대화 모델 이름")
    args = parser.parse_args()

    print("[0/5] 로컬 임베딩 환경을 준비합니다.")
    preparation = ensure_embedding_model(EMBEDDING_MODEL, OLLAMA_URL)
    print(f"      {preparation.message}")
    backend = "ollama" if preparation.ready else "hash"
    if not preparation.ready:
        print("      해싱 대체 모드로 계속 진행합니다.")

    documents, chunks = load_document_chunks(DOCUMENTS_DIR)
    print(f"[1/5] 문서 {len(documents)}개를 읽었습니다.")
    print(f"[2/5] 문서를 {len(chunks)}개 Chunk로 나눴습니다.")

    try:
        embedding_result = create_embeddings(
            [chunk.text for chunk in chunks],
            requested_backend=backend,
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_URL,
        )
    except EmbeddingUnavailable as error:
        print(f"      Ollama 임베딩 생성에 실패했습니다: {error}")
        print("      해싱 대체 모드로 다시 시도합니다.")
        embedding_result = create_embeddings(
            [chunk.text for chunk in chunks],
            requested_backend="hash",
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_URL,
        )
    print(
        f"[3/5] {len(embedding_result.vectors)}개 Embedding을 생성했습니다. "
        f"방식={embedding_result.backend}, 차원={embedding_result.dimensions}"
    )
    save_index(INDEX_PATH, chunks=chunks, embedding_result=embedding_result)
    print(f"[4/5] 검색 색인을 저장했습니다: {INDEX_PATH.relative_to(INDEX_PATH.parent.parent)}")

    index = {
        "embedding": {
            "backend": embedding_result.backend,
            "model": embedding_result.model,
            "dimensions": embedding_result.dimensions,
        },
        "chunks": [
            {**chunk.to_dict(), "vector": vector}
            for chunk, vector in zip(chunks, embedding_result.vectors)
        ],
    }
    results = search_index(
        args.question,
        index=index,
        base_url=OLLAMA_URL,
        top_k=args.top_k,
    )
    print(f"[5/5] 질문과 가까운 규정 {len(results)}개를 찾았습니다.")
    print()
    for number, result in enumerate(results, start=1):
        print(f"{number}위 · 유사도 {result.score:.4f} · {result.heading}")
    print()

    prompt = build_rag_prompt(args.question, results)
    if not args.chat_model:
        print("생성 모델을 지정하지 않아 LLM에 전달할 RAG 입력을 표시합니다.")
        print()
        print(prompt)
        return 0

    try:
        answer = generate_with_ollama(
            prompt,
            model=args.chat_model,
            base_url=OLLAMA_URL,
        )
    except RuntimeError as error:
        print(error)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
