from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mcp import Client, StdioServerParameters, stdio_client

from rag_practice.chunking import load_document_chunks
from rag_practice.config import DOCUMENTS_DIR, INDEX_PATH, ROOT_DIR
from rag_practice.embedding import create_embeddings
from rag_practice.index_store import load_index, save_index
from rag_practice.retrieval import search_index


class RagPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.documents, self.chunks = load_document_chunks(DOCUMENTS_DIR)

    def test_sample_document_contains_thirty_rule_chunks(self) -> None:
        self.assertEqual(len(self.documents), 1)
        self.assertEqual(len(self.chunks), 30)

    def test_hash_search_finds_usb_export_rule(self) -> None:
        result = self.create_hash_embeddings()
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "index.json"
            save_index(index_path, chunks=self.chunks, embedding_result=result)
            index = load_index(index_path)
            matches = search_index(
                "USB를 외부로 반출하려면 어떤 승인이 필요한가?",
                index=index,
                base_url="http://127.0.0.1:9",
                top_k=3,
            )

        self.assertEqual(matches[0].heading, "규칙 13. USB 외부 반출")

    def create_hash_embeddings(self):
        return create_embeddings(
            [chunk.text for chunk in self.chunks],
            requested_backend="hash",
            model="unused",
            base_url="http://127.0.0.1:9",
        )

    def test_auto_backend_falls_back_when_ollama_is_unreachable(self) -> None:
        result = create_embeddings(
            ["보안 담당자의 승인이 필요하다."],
            requested_backend="auto",
            model="embeddinggemma:300m-qat-q4_0",
            base_url="http://127.0.0.1:9",
        )

        self.assertEqual(result.backend, "hash")
        self.assertIsNotNone(result.fallback_reason)
        self.assertEqual(len(result.vectors[0]), 384)

    def test_ollama_backend_reads_embedding_api_response(self) -> None:
        class EmbedHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                content_length = int(self.headers["Content-Length"])
                request_body = json.loads(self.rfile.read(content_length))
                inputs = request_body["input"]
                response_body = json.dumps(
                    {"embeddings": [[1.0, 0.0, 0.0] for _ in inputs]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), EmbedHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = create_embeddings(
                ["첫 번째 문장", "두 번째 문장"],
                requested_backend="ollama",
                model="embeddinggemma:300m-qat-q4_0",
                base_url=f"http://127.0.0.1:{server.server_port}",
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(result.backend, "ollama")
        self.assertEqual(result.dimensions, 3)
        self.assertEqual(len(result.vectors), 2)

    def test_mcp_server_exposes_and_executes_rag_search_tool(self) -> None:
        save_index(
            INDEX_PATH,
            chunks=self.chunks,
            embedding_result=self.create_hash_embeddings(),
        )

        async def call_server() -> tuple[list[str], object]:
            parameters = StdioServerParameters(
                command=sys.executable,
                args=[str(ROOT_DIR / "rag_mcp_server.py")],
                cwd=str(ROOT_DIR),
            )
            async with Client(stdio_client(parameters)) as client:
                tools = await client.list_tools()
                result = await client.call_tool(
                    "search_internal_rules",
                    {
                        "query": "USB를 외부로 반출하려면 누구의 승인이 필요한가?",
                        "top_k": 3,
                    },
                )
                return [tool.name for tool in tools.tools], result.structured_content

        tool_names, structured_content = asyncio.run(call_server())
        self.assertIn("search_internal_rules", tool_names)
        self.assertIsInstance(structured_content, dict)
        self.assertEqual(
            structured_content["matches"][0]["heading"],
            "규칙 13. USB 외부 반출",
        )

    def test_mcp_host_bridges_minimax_tool_call_to_rag_server(self) -> None:
        save_index(
            INDEX_PATH,
            chunks=self.chunks,
            embedding_result=self.create_hash_embeddings(),
        )
        chat_requests = []

        class OllamaHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                response_body = json.dumps(
                    {"models": [{"name": "minimax-m3:cloud"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def do_POST(self) -> None:
                content_length = int(self.headers["Content-Length"])
                request_body = json.loads(self.rfile.read(content_length))
                chat_requests.append(request_body)
                if len(chat_requests) == 1:
                    message = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "search_internal_rules",
                                    "arguments": {
                                        "query": "USB 외부 반출 승인",
                                        "top_k": 3,
                                    },
                                },
                            }
                        ],
                    }
                else:
                    message = {
                        "role": "assistant",
                        "content": (
                            "정보 보안 담당자의 사전 승인이 필요합니다. "
                            "[규칙 13. USB 외부 반출]"
                        ),
                    }
                response_body = json.dumps(
                    {"message": message},
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), OllamaHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        environment = os.environ.copy()
        environment["RAG_OLLAMA_URL"] = f"http://127.0.0.1:{server.server_port}"
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT_DIR / "rag_chat.py"),
                    "--question",
                    "USB를 외부로 반출하려면 누구의 승인이 필요한가?",
                ],
                cwd=ROOT_DIR,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("[MCP] search_internal_rules 호출", completed.stdout)
        self.assertIn("정보 보안 담당자의 사전 승인", completed.stdout)
        self.assertEqual(len(chat_requests), 2)
        self.assertTrue(chat_requests[0]["tools"])
        self.assertEqual(chat_requests[1]["messages"][-1]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
