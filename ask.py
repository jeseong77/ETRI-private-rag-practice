from __future__ import annotations

import argparse

from rag_practice.config import DEFAULT_TOP_K, INDEX_PATH, OLLAMA_URL
from rag_practice.generation import build_rag_prompt, generate_with_ollama
from rag_practice.index_store import load_index
from rag_practice.retrieval import search_index


def main() -> int:
    parser = argparse.ArgumentParser(description="검색 근거로 RAG 입력 또는 답변 생성")
    parser.add_argument("question", help="내부 규정에 묻고 싶은 질문")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--chat-model",
        help="설치된 Ollama 대화 모델 이름. 생략하면 LLM에 전달할 입력만 출력",
    )
    args = parser.parse_args()

    index = load_index(INDEX_PATH)
    results = search_index(
        args.question,
        index=index,
        base_url=OLLAMA_URL,
        top_k=args.top_k,
    )
    prompt = build_rag_prompt(args.question, results)

    if not args.chat_model:
        print("생성 모델을 사용하지 않고 LLM에 전달할 RAG 입력을 표시합니다.")
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
