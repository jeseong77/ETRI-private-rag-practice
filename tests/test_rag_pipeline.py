from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from mcp import Client, StdioServerParameters, stdio_client

from configure_claude_mcp import SERVER_NAME, SERVER_PATH, registration_command
from rag_practice.chunking import load_document_chunks
from rag_practice.config import DOCUMENTS_DIR, INDEX_PATH, ROOT_DIR
from rag_practice.embedding import create_hash_embeddings
from rag_practice.index_store import load_index, save_index
from rag_practice.retrieval import search_index


class RagPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.documents, self.chunks = load_document_chunks(DOCUMENTS_DIR)

    def create_index(self, path: Path) -> None:
        save_index(
            path,
            chunks=self.chunks,
            embedding_result=create_hash_embeddings(
                [chunk.text for chunk in self.chunks]
            ),
        )

    def test_sample_document_contains_thirty_rule_chunks(self) -> None:
        self.assertEqual(len(self.documents), 1)
        self.assertEqual(len(self.chunks), 30)

    def test_hash_search_finds_usb_export_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "index.json"
            self.create_index(index_path)
            index = load_index(index_path)
            matches = search_index(
                "USB를 외부로 반출하려면 어떤 승인이 필요한가?",
                index=index,
                top_k=3,
            )

        self.assertEqual(matches[0].heading, "규칙 13. USB 외부 반출")

    def test_claude_registration_uses_current_virtual_environment(self) -> None:
        command = registration_command("claude")

        self.assertEqual(command[0], "claude")
        self.assertIn("local", command)
        self.assertIn(SERVER_NAME, command)
        self.assertIn(sys.executable, command)
        self.assertIn(str(SERVER_PATH), command)

    def test_mcp_server_exposes_and_executes_rag_search_tool(self) -> None:
        self.create_index(INDEX_PATH)

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


if __name__ == "__main__":
    unittest.main()
