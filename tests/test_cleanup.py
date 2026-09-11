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
    assert pipeline.clean_transcript("model", "\n\n".join([first, last])) == "\n\n".join([first, last])


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


def test_clean_transcript_preserves_source_after_empty_body_or_severe_contraction(monkeypatch):
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: "[00:00:00 - 00:00:01] ")
    source = "[00:00:00 - 00:00:01] raw transcript"
    assert pipeline.clean_transcript("model", source) == source

    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: "[00:00:00 - 00:00:01] We discussed sensors.")
    source = "[00:00:00 - 00:00:01] We discussed sensors. The red sensor trips at 17 volts and the blue sensor resets after 23 seconds."
    assert pipeline.clean_transcript("model", source) == source


@pytest.mark.parametrize("source, replacement", [
    ("可以找小林确认。", "可以找 algorithm 确认。"),
    ("Testing, one, two.", "测试，一，二。"),
])
def test_cleanup_rejects_introducing_a_language_absent_from_the_segment(monkeypatch, source, replacement):
    label = "[00:00:00 - 00:00:01] "
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: label + replacement)
    with pytest.raises(RuntimeError, match="segment language changed"):
        pipeline._validate_cleaned_chunk(label + source, label + replacement)
    assert pipeline.clean_transcript("model", label + source) == label + source


def test_cleanup_retries_language_changed_group_in_smaller_chunks(monkeypatch):
    first = "[00:00:00 - 00:00:01] 可以找小林确认。"
    last = "[00:00:01 - 00:00:02] 请保持服务稳定。"
    source = first + "\n\n" + last
    responses = iter([source.replace("小林", "algorithm"), first, last])
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: next(responses))
    assert pipeline.clean_transcript("model", source) == source


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
    monkeypatch.setattr("voicenotes.ollama.time.sleep", lambda seconds: None)
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


def test_generate_retries_incomplete_response_without_returning_partial_text(monkeypatch):
    responses = iter([{"response": "partial", "done": False}, {"response": "complete", "done": True}])
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse(next(responses)))
    monkeypatch.setattr("voicenotes.ollama.time.sleep", lambda seconds: None)
    assert generate("model", "prompt") == "complete"


def test_generate_incomplete_retries_are_bounded_and_diagnostic(monkeypatch):
    calls = []

    def incomplete(*args, **kwargs):
        calls.append(1)
        return FakeResponse({"response": "partial", "done": False})

    monkeypatch.setattr("urllib.request.urlopen", incomplete)
    monkeypatch.setattr("voicenotes.ollama.time.sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="did not complete.*done=False.*response_chars=7"):
        generate("model", "prompt")
    assert len(calls) == 3


def test_generate_retries_transport_failure_but_not_missing_model(monkeypatch):
    from urllib.error import HTTPError, URLError
    responses = iter([URLError("connection lost"), {"response": "complete", "done": True}])

    def response(*args, **kwargs):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)

    monkeypatch.setattr("urllib.request.urlopen", response)
    monkeypatch.setattr("voicenotes.ollama.time.sleep", lambda seconds: None)
    assert generate("model", "prompt") == "complete"
    calls = []

    def missing(*args, **kwargs):
        calls.append(1)
        raise HTTPError("local", 404, "model missing", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", missing)
    with pytest.raises(HTTPError):
        generate("model", "prompt")
    assert len(calls) == 1


def test_cleanup_resumes_validated_recursive_chunks_after_incomplete_generation(tmp_path, monkeypatch):
    first = "[00:00:00 - 00:00:01] first"
    last = "[00:00:01 - 00:00:02] last"
    raw = first + "\n\n" + last
    calls = []

    def interrupted(model, prompt, **kwargs):
        source = prompt.rsplit("Transcript:\n", 1)[1].strip()
        calls.append(source)
        if source == raw:
            raise pipeline.ollama.IncompleteGenerationError("interrupted")
        if source == last:
            raise RuntimeError("server unavailable")
        return source

    monkeypatch.setattr("voicenotes.ollama.generate", interrupted)
    with pytest.raises(RuntimeError, match="server unavailable"):
        pipeline.clean_transcript("model", raw, session=tmp_path)
    assert calls == [raw, first, last]
    assert not (tmp_path / "transcript_clean.md").exists()
    calls.clear()

    def resumed(model, prompt, **kwargs):
        source = prompt.rsplit("Transcript:\n", 1)[1].strip()
        calls.append(source)
        return source

    monkeypatch.setattr("voicenotes.ollama.generate", resumed)
    assert pipeline.clean_transcript("model", raw, session=tmp_path) == raw
    assert calls == [last]
    assert "reused" in (tmp_path / "pipeline.log").read_text()


@pytest.mark.parametrize("change", ["model", "prompt", "source", "corrupt"])
def test_cleanup_checkpoint_never_reuses_stale_or_invalid_output(tmp_path, monkeypatch, change):
    raw = "[00:00:00 - 00:00:01] first"
    calls = []

    def response(model, prompt, **kwargs):
        calls.append(prompt)
        return prompt.rsplit("Transcript:\n", 1)[1].strip()

    monkeypatch.setattr("voicenotes.ollama.generate", response)
    pipeline.clean_transcript("model", raw, session=tmp_path)
    calls.clear()
    model = "model"
    if change == "model":
        model = "different"
    elif change == "prompt":
        monkeypatch.setattr(pipeline, "CLEANUP_PROMPT", "Updated instructions.\n" + pipeline.CLEANUP_PROMPT)
    elif change == "source":
        raw = raw.replace("first", "second")
    else:
        cache = tmp_path / ".generation-cache.json"
        data = json.loads(cache.read_text())
        cache.write_text(json.dumps({key: "invalid shortened output" for key in data}))
    assert pipeline.clean_transcript(model, raw, session=tmp_path) == raw
    assert len(calls) == 1


def test_cleanup_checkpoint_surrounding_whitespace_cannot_drop_a_paragraph(tmp_path, monkeypatch):
    raw = "[00:00:00 - 00:00:01] first\n\n[00:00:01 - 00:00:02] last"
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: raw)
    assert pipeline.clean_transcript("model", raw, session=tmp_path) == raw
    cache = tmp_path / ".generation-cache.json"
    data = json.loads(cache.read_text())
    cache.write_text(json.dumps({key: "\n\n" + value + "\n\n" for key, value in data.items()}))
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: pytest.fail("reuse validated cache"))
    assert pipeline.clean_transcript("model", raw, session=tmp_path) == raw


def test_cleanup_token_limit_splits_group_without_accepting_partial_output(monkeypatch):
    raw = "[00:00:00 - 00:00:01] first\n\n[00:00:01 - 00:00:02] last"

    def response(model, prompt, **kwargs):
        source = prompt.rsplit("Transcript:\n", 1)[1].strip()
        if source == raw:
            raise pipeline.ollama.OutputLimitError("token limit")
        return source

    monkeypatch.setattr("voicenotes.ollama.generate", response)
    assert pipeline.clean_transcript("model", raw) == raw


def test_cleanup_exhausted_single_paragraph_fails_without_checkpoint(tmp_path, monkeypatch):
    def response(*args, **kwargs):
        raise pipeline.ollama.IncompleteGenerationError("interrupted")

    monkeypatch.setattr("voicenotes.ollama.generate", response)
    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline.clean_transcript("model", "[00:00:00 - 00:00:01] first", session=tmp_path)
    assert not (tmp_path / ".generation-cache.json").exists()


def test_single_paragraph_validation_fallback_is_logged_and_checkpointed(tmp_path, monkeypatch):
    source = "[00:00:00 - 00:00:01] Keep the source language."
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: "[00:00:00 - 00:00:01] 保持原语言。")
    assert pipeline.clean_transcript("model", source, session=tmp_path) == source
    assert "retained original paragraph" in (tmp_path / "pipeline.log").read_text()
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: pytest.fail("reuse checkpoint"))
    assert pipeline.clean_transcript("model", source, session=tmp_path) == source


def test_single_paragraph_fallback_cannot_accept_malformed_raw_input(monkeypatch):
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: "invalid")
    with pytest.raises(RuntimeError, match="timestamps missing or reordered"):
        pipeline.clean_transcript("model", "malformed source without a timestamp")
