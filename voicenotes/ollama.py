from __future__ import annotations

import json
import subprocess
import time
from urllib.error import URLError
import urllib.request


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.2
OLLAMA_CONTEXT_LENGTH = 8192
OLLAMA_KEEP_ALIVE = "30s"
OLLAMA_TIMEOUT_SECONDS = 1800
OLLAMA_START_TIMEOUT_SECONDS = 60


def _read_tags() -> dict[str, object]:
    with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=10) as response:
        return dict(json.loads(response.read().decode("utf-8")))


def _wait_for_server(timeout_seconds: int = OLLAMA_START_TIMEOUT_SECONDS) -> dict[str, object]:
    subprocess.run(["open", "-ga", "Ollama"], check=False)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            return _read_tags()
        except URLError:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for Ollama server to start. Open Ollama and retry.")
            time.sleep(1)


def ensure_model_available(model: str) -> None:
    try:
        payload = _read_tags()
    except URLError:
        payload = _wait_for_server()
    names = {item.get("name") for item in payload.get("models", [])}
    if model not in names:
        raise RuntimeError(f"Ollama model missing. Run: ollama pull {model}")


def generate(model: str, prompt: str, timeout_seconds: int = OLLAMA_TIMEOUT_SECONDS, max_output_tokens: int | None = None, system_prompt: str | None = None) -> str:
    options: dict[str, object] = {"temperature": OLLAMA_TEMPERATURE, "num_ctx": OLLAMA_CONTEXT_LENGTH, "presence_penalty": 0}
    if max_output_tokens is not None:
        options["num_predict"] = max_output_tokens
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": options,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    if system_prompt is not None:
        payload["system"] = system_prompt
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    if body.get("done") is not True:
        raise RuntimeError("Ollama generation did not complete")
    if body.get("done_reason") == "length":
        raise RuntimeError("Ollama generation stopped at its token limit")
    generated = str(body.get("response", ""))
    if not generated.strip():
        raise RuntimeError("Ollama returned a blank response")
    return generated
