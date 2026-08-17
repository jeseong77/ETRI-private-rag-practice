from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    heading: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def chunk_markdown(path: Path) -> list[Chunk]:
    """Markdown의 두 번째 단계 제목 하나를 Chunk 하나로 만든다."""

    chunks: list[Chunk] = []
    heading = ""
    body: list[str] = []

    def flush() -> None:
        if not heading:
            return

        content = "\n".join(line for line in body if line).strip()
        chunk_number = len(chunks) + 1
        chunks.append(
            Chunk(
                id=f"{path.stem}-{chunk_number:03d}",
                source=path.name,
                heading=heading,
                text=f"{heading}\n{content}".strip(),
            )
        )

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            body = []
            continue
        if heading:
            body.append(line)

    flush()
    return chunks


def load_document_chunks(documents_dir: Path) -> tuple[list[Path], list[Chunk]]:
    documents = sorted(documents_dir.glob("*.md"))
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_markdown(document))
    return documents, chunks
