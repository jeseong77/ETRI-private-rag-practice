from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SERVER_NAME = "internal-rules-rag"
ROOT_DIR = Path(__file__).resolve().parent
SERVER_PATH = ROOT_DIR / "rag_mcp_server.py"


def registration_command(claude_executable: str) -> list[str]:
    return [
        claude_executable,
        "mcp",
        "add",
        "--transport",
        "stdio",
        "--scope",
        "local",
        SERVER_NAME,
        "--",
        sys.executable,
        str(SERVER_PATH),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="현재 Python 가상 환경으로 Claude Code MCP Server 등록"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Claude Code 설정을 바꾸지 않고 등록 명령만 출력",
    )
    args = parser.parse_args()

    claude_executable = shutil.which("claude")
    if claude_executable is None:
        print("Claude Code 명령을 찾지 못했습니다. Claude Code를 먼저 설치하세요.")
        return 1
    if not SERVER_PATH.exists():
        print(f"MCP Server 파일을 찾지 못했습니다: {SERVER_PATH}")
        return 1

    command = registration_command(claude_executable)
    print(f"Claude Code: {claude_executable}")
    print(f"Python: {sys.executable}")
    print(f"MCP Server: {SERVER_PATH}")
    if args.dry_run:
        print("등록 명령:")
        print(" ".join(command))
        return 0

    subprocess.run(
        [claude_executable, "mcp", "remove", SERVER_NAME, "--scope", "local"],
        cwd=ROOT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    completed = subprocess.run(command, cwd=ROOT_DIR, check=False)
    if completed.returncode != 0:
        print("Claude Code MCP Server 등록에 실패했습니다.")
        return completed.returncode

    subprocess.run(
        [claude_executable, "mcp", "get", SERVER_NAME],
        cwd=ROOT_DIR,
        check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
