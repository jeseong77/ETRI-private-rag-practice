from __future__ import annotations

from rag_practice.chunking import load_document_chunks
from rag_practice.config import DOCUMENTS_DIR, INDEX_PATH
from rag_practice.embedding import create_hash_embeddings
from rag_practice.index_store import save_index


def main() -> int:
    documents, chunks = load_document_chunks(DOCUMENTS_DIR)
    print(f"[1/4] Markdown 문서 {len(documents)}개를 읽었습니다.")
    print(f"[2/4] 문서를 {len(chunks)}개 Chunk로 나눴습니다.")
    if not chunks:
        print("색인할 Chunk가 없습니다.")
        return 1

    embedding_result = create_hash_embeddings([chunk.text for chunk in chunks])

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
