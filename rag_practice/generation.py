from __future__ import annotations

import json
import urllib.error
import urllib.request

from rag_practice.retrieval import SearchResult


def build_rag_prompt(question: str, results: list[SearchResult]) -> str:
    evidence = "\n\n".join(
        f"[근거 {number}]\n출처: {result.source}\n{result.text}"
        for number, result in enumerate(results, start=1)
    )
    return (
        "아래 근거만 사용해 질문에 답하세요. 근거가 부족하면 부족하다고 말하세요. "
        "답변 끝에는 사용한 근거 번호를 표시하세요.\n\n"
        f"질문: {question}\n\n{evidence}"
    )


def generate_with_ollama(
    prompt: str,
    *,
    model: str,
    base_url: str,
    timeout_seconds: float = 120.0,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Ollama 답변 생성 실패: {error}") from error

    content = payload.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama 답변에 텍스트가 없습니다.")
    return content.strip()
