from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters, Tool, stdio_client
from mcp.types import CallToolResult

from rag_practice.bootstrap import ensure_model
from rag_practice.config import DEFAULT_TOP_K, INDEX_PATH, OLLAMA_URL
from rag_practice.index_store import load_index
from rag_practice.ollama_chat import OllamaChatUnavailable, chat


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = os.environ.get("RAG_CHAT_MODEL", "minimax-m3:cloud")
SEARCH_TOOL_NAME = "search_internal_rules"
SYSTEM_MESSAGE = (
    "당신은 내부 규정 질의 도우미입니다. 내부 규정에 관한 질문에는 반드시 "
    "search_internal_rules 도구를 먼저 사용하세요. 도구가 반환한 원문만 근거로 "
    "답하고 답변 끝에 규정 제목과 출처를 표시하세요. 근거가 부족하면 추측하지 마세요."
)


def to_ollama_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def tool_result_text(result: CallToolResult) -> str:
    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False, indent=2)

    texts = []
    for content in result.content:
        text = getattr(content, "text", None)
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)


async def call_search_tool(client: Client, query: str, top_k: int) -> str:
    print(f"[MCP] {SEARCH_TOOL_NAME} 호출: query={query!r}, top_k={top_k}")
    result = await client.call_tool(
        SEARCH_TOOL_NAME,
        {"query": query, "top_k": top_k},
    )
    if result.is_error:
        raise RuntimeError(f"MCP Tool 실행 실패: {tool_result_text(result)}")
    text = tool_result_text(result)
    print("[MCP] 검색 결과를 받았습니다.")
    return text


async def answer_question(
    *,
    client: Client,
    question: str,
    model: str,
    top_k: int,
    ollama_tools: list[dict[str, Any]],
) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": question},
    ]

    try:
        assistant = chat(
            model=model,
            messages=messages,
            tools=ollama_tools,
            base_url=OLLAMA_URL,
        )
    except OllamaChatUnavailable as error:
        print(f"[Ollama] {error}")
        print("[대체 동작] MCP 검색 결과를 그대로 표시합니다.")
        return await call_search_tool(client, question, top_k)

    messages.append(assistant)
    tool_calls = assistant.get("tool_calls") or []
    if not tool_calls:
        print("[Host] 모델이 Tool을 호출하지 않아 RAG 검색을 직접 실행합니다.")
        evidence = await call_search_tool(client, question, top_k)
        fallback_messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": (
                    f"질문: {question}\n\nMCP 검색 결과:\n{evidence}\n\n"
                    "검색 결과만 사용해 답하세요."
                ),
            },
        ]
        final_message = chat(
            model=model,
            messages=fallback_messages,
            tools=None,
            base_url=OLLAMA_URL,
        )
        return str(final_message.get("content") or "")

    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments") or {}
        if name != SEARCH_TOOL_NAME:
            result_text = f"지원하지 않는 Tool입니다: {name}"
        else:
            result_text = await call_search_tool(
                client,
                str(arguments.get("query") or question),
                int(arguments.get("top_k") or top_k),
            )
        messages.append(
            {
                "role": "tool",
                "tool_name": name,
                "content": result_text,
            }
        )

    final_message = chat(
        model=model,
        messages=messages,
        tools=ollama_tools,
        base_url=OLLAMA_URL,
    )
    return str(final_message.get("content") or "")


async def run_chat(model: str, question: str | None, top_k: int) -> int:
    if not INDEX_PATH.exists():
        print("검색 색인이 없습니다. 먼저 python index_documents.py를 실행하세요.")
        return 1

    index = load_index(INDEX_PATH)
    embedding = index["embedding"]
    print(f"[색인] {len(index['chunks'])}개 Chunk / {embedding['backend']} / {embedding['model']}")

    preparation = ensure_model(model, OLLAMA_URL, "Chat")
    if preparation.ready:
        print(f"[Ollama] {preparation.message}")
    else:
        print(f"[Ollama] {preparation.message}")
        print("         질문 시 MCP 검색 결과만 표시합니다.")

    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT_DIR / "rag_mcp_server.py")],
        cwd=str(ROOT_DIR),
    )
    async with Client(stdio_client(server_parameters)) as client:
        listed_tools = await client.list_tools()
        if not any(tool.name == SEARCH_TOOL_NAME for tool in listed_tools.tools):
            print(f"MCP Server가 {SEARCH_TOOL_NAME} Tool을 제공하지 않습니다.")
            return 1

        print(f"[MCP] 연결됨 / Tool={SEARCH_TOOL_NAME}")
        ollama_tools = to_ollama_tools(listed_tools.tools)

        if question:
            print(f"질문> {question}")
            answer = await answer_question(
                client=client,
                question=question,
                model=model,
                top_k=top_k,
                ollama_tools=ollama_tools,
            )
            print(f"답변> {answer}")
            return 0

        print("질문을 입력하세요. 종료하려면 exit를 입력하세요.")
        while True:
            user_input = input("질문> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                return 0
            if not user_input:
                continue
            answer = await answer_question(
                client=client,
                question=user_input,
                model=model,
                top_k=top_k,
                ollama_tools=ollama_tools,
            )
            print(f"답변> {answer}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama와 RAG MCP Server를 연결한 채팅")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--question", help="한 번 질문한 뒤 종료")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    return asyncio.run(run_chat(args.model, args.question, args.top_k))


if __name__ == "__main__":
    raise SystemExit(main())
