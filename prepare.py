from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.error
import urllib.request

from rag_practice.config import EMBEDDING_MODEL, OLLAMA_URL


def ollama_is_running() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2):
            return True
    except (urllib.error.URLError, TimeoutError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama와 임베딩 모델 준비 상태 확인")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="모델을 내려받지 않고 현재 상태만 확인",
    )
    args = parser.parse_args()

    print("[1/3] Python 실행 환경을 확인했습니다.")
    if shutil.which("ollama") is None:
        print("[2/3] Ollama 명령을 찾지 못했습니다.")
        print("[3/3] 해싱 대체 모드로 실습을 진행할 수 있습니다.")
        return 0

    print("[2/3] Ollama가 설치되어 있습니다.")
    if not ollama_is_running():
        print("[3/3] Ollama가 실행 중이 아닙니다.")
        print("      Ollama를 실행하거나 해싱 대체 모드로 실습하세요.")
        return 0

    if args.skip_download:
        print("[3/3] 모델 다운로드를 건너뛰었습니다.")
        return 0

    print(f"[3/3] 임베딩 모델을 준비합니다: {EMBEDDING_MODEL}")
    completed = subprocess.run(["ollama", "pull", EMBEDDING_MODEL], check=False)
    if completed.returncode != 0:
        print("[안내] 모델 다운로드에 실패했습니다.")
        print("       python ingest.py를 실행하면 해싱 대체 모드가 선택됩니다.")
        return 0

    print("임베딩 모델 준비 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

