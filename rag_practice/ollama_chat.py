from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OllamaChatUnavailable(RuntimeError):
    """Ollama 대화 API를 사용할 수 없을 때 발생한다."""


def chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    base_url: str,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise OllamaChatUnavailable(
            f"Ollama가 요청을 거절했습니다: HTTP {error.code} {detail}"
        ) from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise OllamaChatUnavailable(f"Ollama 대화 요청 실패: {error}") from error

    message = body.get("message")
    if not isinstance(message, dict):
        raise OllamaChatUnavailable("Ollama 응답에 assistant message가 없습니다.")
    return message
