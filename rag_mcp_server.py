from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from rag_practice.config import DEFAULT_TOP_K, INDEX_PATH
from rag_practice.index_store import load_index
from rag_practice.retrieval import search_index


server = MCPServer(
    name="internal-rules-rag",
    title="내부 규정 RAG 검색",
    description="색인된 내부 규정에서 질문과 관련된 원문을 찾습니다.",
)


@server.tool(structured_output=True)
def search_internal_rules(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """내부 규정에서 질문과 관련된 원문을 유사도 순서로 검색한다.

    Args:
        query: 내부 규정에 관해 찾고 싶은 자연어 질문.
        top_k: 반환할 규정 원문 수. 기본값은 3.
    """

    safe_top_k = max(1, min(top_k, 5))
    index = load_index(INDEX_PATH)
    results = search_index(
        query,
        index=index,
        top_k=safe_top_k,
    )
    return {
        "query": query,
        "embedding": index["embedding"],
        "matches": [
            {
                "rank": rank,
                "score": round(result.score, 4),
                "source": result.source,
                "heading": result.heading,
                "text": result.text,
            }
            for rank, result in enumerate(results, start=1)
        ],
    }


if __name__ == "__main__":
    server.run(transport="stdio")
