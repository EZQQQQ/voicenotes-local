import json
from urllib.error import URLError

import pytest

from voicenotes.ollama import ensure_model_available, generate


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_ensure_model_available_accepts_configured_model(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=10: FakeResponse({"models": [{"name": "qwen2.5:14b"}]}))

    ensure_model_available("qwen2.5:14b")


def test_ensure_model_available_starts_ollama_when_server_is_closed(monkeypatch):
    calls = iter([URLError("closed"), URLError("closed"), FakeResponse({"models": [{"name": "qwen2.5:14b"}]})])
    opened = []
    slept = []

    def fake_urlopen(request, timeout=10):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("subprocess.run", lambda args, check=False: opened.append(args))
    monkeypatch.setattr("time.sleep", lambda seconds: slept.append(seconds))

    ensure_model_available("qwen2.5:14b")

    assert opened == [["open", "-ga", "Ollama"]]
    assert slept == [1]


def test_ensure_model_available_fails_with_pull_instruction(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=10: FakeResponse({"models": []}))

    with pytest.raises(RuntimeError, match="ollama pull qwen2.5:14b"):
        ensure_model_available("qwen2.5:14b")


def test_generate_posts_non_streaming_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse({"response": "clean transcript"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = generate("qwen2.5:14b", "Prompt text", timeout_seconds=1800)

    assert response == "clean transcript"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["payload"] == {
        "model": "qwen2.5:14b",
        "prompt": "Prompt text",
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
        "keep_alive": "30s",
    }
    assert captured["timeout"] == 1800
