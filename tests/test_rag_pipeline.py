from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rag_practice.chunking import load_document_chunks
from rag_practice.config import DOCUMENTS_DIR
from rag_practice.embedding import create_embeddings
from rag_practice.generation import build_rag_prompt, generate_with_ollama
from rag_practice.index_store import load_index, save_index
from rag_practice.retrieval import SearchResult, search_index


class RagPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.documents, self.chunks = load_document_chunks(DOCUMENTS_DIR)

    def test_sample_document_contains_thirty_rule_chunks(self) -> None:
        self.assertEqual(len(self.documents), 1)
        self.assertEqual(len(self.chunks), 30)

    def test_hash_search_finds_usb_export_rule(self) -> None:
        result = create_embeddings(
            [chunk.text for chunk in self.chunks],
            requested_backend="hash",
            model="unused",
            base_url="http://127.0.0.1:9",
        )
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

    def test_rag_prompt_is_sent_to_ollama_chat_api(self) -> None:
        received_prompt = []

        class ChatHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                content_length = int(self.headers["Content-Length"])
                request_body = json.loads(self.rfile.read(content_length))
                received_prompt.append(request_body["messages"][0]["content"])
                response_body = json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "content": "정보 보안 담당자의 사전 승인이 필요합니다. [근거 1]",
                        }
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, format: str, *args: object) -> None:
                return

        prompt = build_rag_prompt(
            "USB 반출에 필요한 승인은?",
            [
                SearchResult(
                    score=0.9,
                    source="규정.md",
                    heading="규칙 13. USB 외부 반출",
                    text="정보 보안 담당자의 사전 승인을 받아야 한다.",
                )
            ],
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            answer = generate_with_ollama(
                prompt,
                model="local-chat-model",
                base_url=f"http://127.0.0.1:{server.server_port}",
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertIn("정보 보안 담당자", answer)
        self.assertIn("규칙 13. USB 외부 반출", received_prompt[0])
        self.assertIn("USB 반출에 필요한 승인은?", received_prompt[0])


if __name__ == "__main__":
    unittest.main()
