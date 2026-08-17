from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreparation:
    ready: bool
    message: str


def _read_json(url: str, timeout_seconds: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_is_running(base_url: str) -> bool:
    try:
        _read_json(f"{base_url.rstrip('/')}/api/tags", timeout_seconds=2)
        return True
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False


def start_local_ollama(base_url: str) -> bool:
    if ollama_is_running(base_url):
        return True
    if shutil.which("ollama") is None:
        return False

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(20):
        if ollama_is_running(base_url):
            return True
        time.sleep(0.25)
    return False


def installed_models(base_url: str) -> set[str]:
    payload = _read_json(f"{base_url.rstrip('/')}/api/tags", timeout_seconds=5)
    names = set()
    for model in payload.get("models", []):
        name = model.get("name") or model.get("model")
        if isinstance(name, str):
            names.add(name)
    return names


def pull_model(model: str, base_url: str) -> None:
    body = json.dumps({"model": model, "stream": True}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/pull",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_status = ""
    last_percent = -10
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line.decode("utf-8"))
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))

            status = str(payload.get("status", ""))
            total = int(payload.get("total") or 0)
            completed = int(payload.get("completed") or 0)
            percent = int(completed * 100 / total) if total else -1

            if percent >= 0 and percent >= last_percent + 10:
                print(f"      {status} {percent}%")
                last_percent = percent
            elif status and status != last_status and percent < 0:
                print(f"      {status}")
            last_status = status


def ensure_embedding_model(model: str, base_url: str) -> ModelPreparation:
    if not start_local_ollama(base_url):
        if shutil.which("ollama") is None:
            return ModelPreparation(
                ready=False,
                message="Ollama 프로그램을 찾지 못했습니다.",
            )
        return ModelPreparation(
            ready=False,
            message="Ollama를 실행하지 못했습니다.",
        )

    try:
        models = installed_models(base_url)
        if model in models:
            return ModelPreparation(
                ready=True,
                message=f"임베딩 모델이 준비되어 있습니다: {model}",
            )

        print(f"[준비] 임베딩 모델을 내려받습니다: {model}")
        pull_model(model, base_url)
        return ModelPreparation(
            ready=True,
            message=f"임베딩 모델 다운로드를 완료했습니다: {model}",
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as error:
        return ModelPreparation(
            ready=False,
            message=f"임베딩 모델 다운로드에 실패했습니다: {error}",
        )
