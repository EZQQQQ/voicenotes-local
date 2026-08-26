from __future__ import annotations

import json
import urllib.request


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.2
OLLAMA_KEEP_ALIVE = "30s"
OLLAMA_TIMEOUT_SECONDS = 1800


def ensure_model_available(model: str) -> None:
    with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    names = {item.get("name") for item in payload.get("models", [])}
    if model not in names:
        raise RuntimeError(f"Ollama model missing. Run: ollama pull {model}")


def generate(model: str, prompt: str, timeout_seconds: int = OLLAMA_TIMEOUT_SECONDS) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": OLLAMA_TEMPERATURE},
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("response", ""))
