import json

import pytest

from voicenotes import pipeline
from voicenotes.ollama import generate


def test_estimate_tokens_and_chunk_paragraphs_preserve_paragraph_boundaries():
    assert pipeline.estimate_tokens("你好hello") == 4
    assert pipeline.chunk_paragraphs(["A" * 8, "B" * 8], 5) == [["A" * 8, "B" * 8]]
    assert pipeline.chunk_paragraphs(["A" * 8, "B" * 8], 4) == [["A" * 8], ["B" * 8]]
    with pytest.raises(ValueError, match="paragraph exceeds chunk budget"):
        pipeline.chunk_paragraphs(["X" * 200], 25)


def test_collapse_filler_runs_only_collapses_three_close_identical_recognized_fillers():
    assert pipeline.collapse_filler_runs([
        "[00:00:01 - 00:00:02] 嗯。",
        "[00:00:02 - 00:00:03] 嗯",
        "[00:00:03 - 00:00:04] 嗯",
    ]) == ["[00:00:01 - 00:00:04] 嗯。"]
    preserved = [
        "[00:00:01 - 00:00:02] 嗯",
        "[00:00:10 - 00:00:11] 嗯",
        "[00:00:20 - 00:00:21] 嗯",
    ]
    assert pipeline.collapse_filler_runs(preserved) == preserved


def test_clean_transcript_preserves_blank_paragraphs_and_validates_timestamp_order(monkeypatch):
    first = "[00:00:00 - 00:00:01] raw first"
    blank = "[00:00:01 - 00:00:01]"
    last = "[00:00:01 - 00:00:02] raw last"
    raw = "\n\n".join([first, blank, blank, last])
    calls = []

    def fake_generate(model, prompt, **kwargs):
        source = prompt.rsplit("Transcript:\n", 1)[1].strip()
        calls.append(source)
        return source.replace("raw", "cleaned")

    monkeypatch.setattr("voicenotes.ollama.generate", fake_generate)
    assert pipeline.clean_transcript("model", raw) == "\n\n".join([first.replace("raw", "cleaned"), blank, blank, last.replace("raw", "cleaned")])
    assert calls == ["\n\n".join([first, last])]

    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: first)
    with pytest.raises(RuntimeError, match="timestamps missing or reordered"):
        pipeline.clean_transcript("model", "\n\n".join([first, last]))


def test_clean_transcript_bisects_invalid_completed_group_but_propagates_generation_errors(monkeypatch):
    raw = "\n\n".join([
        "[00:00:00 - 00:00:01] first",
        "[00:00:01 - 00:00:02] second",
    ])
    responses = iter([
        "[00:00:00 - 00:00:01] first",
        "[00:00:00 - 00:00:01] first",
        "[00:00:01 - 00:00:02] second",
    ])
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: next(responses))
    assert pipeline.clean_transcript("model", raw) == raw

    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("connection lost")))
    with pytest.raises(RuntimeError, match="connection lost"):
        pipeline.clean_transcript("model", raw)


def test_clean_transcript_rejects_empty_body_and_severe_contraction(monkeypatch):
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: "[00:00:00 - 00:00:01] ")
    with pytest.raises(RuntimeError, match="segment content missing"):
        pipeline.clean_transcript("model", "[00:00:00 - 00:00:01] raw transcript")

    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: "[00:00:00 - 00:00:01] We discussed sensors.")
    with pytest.raises(RuntimeError, match="segment content shortened"):
        pipeline.clean_transcript("model", "[00:00:00 - 00:00:01] We discussed sensors. The red sensor trips at 17 volts and the blue sensor resets after 23 seconds.")


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_generate_sets_num_predict_only_when_requested_and_rejects_bad_completion(monkeypatch):
    captured = []

    def fake_urlopen(request, timeout):
        captured.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse({"response": "clean", "done": True, "done_reason": "stop"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert generate("model", "prompt") == "clean"
    assert generate("model", "prompt", max_output_tokens=123) == "clean"
    assert captured[0]["options"] == {"temperature": 0.2, "num_ctx": 8192, "presence_penalty": 0}
    assert captured[1]["options"]["num_predict"] == 123

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse({"response": "", "thinking": "internal reasoning only", "done": True, "done_reason": "stop"}))
    with pytest.raises(RuntimeError, match="blank response"):
        generate("model", "prompt")

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse({"response": "partial", "done": False}))
    with pytest.raises(RuntimeError, match="did not complete"):
        generate("model", "prompt")

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse({"response": "partial", "done": True, "done_reason": "length"}))
    with pytest.raises(RuntimeError, match="token limit"):
        generate("model", "prompt")
