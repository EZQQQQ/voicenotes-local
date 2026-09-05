# Cleanup Context-Overflow & ASR Filler Noise Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make long or noisy recordings complete without silent transcript loss by bounding every cleanup and summary request, failing closed on truncated model output, and preserving the verbatim ASR transcript for audit and retry.

**Architecture:** Keep `transcript_raw.md` as the exact timestamped output from Whisper. Build a derived cleanup input that conservatively collapses only dense runs of the same filler token, split that input at paragraph boundaries into bounded requests, and reject any cleanup response that is incomplete or loses/reorders timestamps. Preserve the existing 8K Ollama context chosen for 16GB Macs. Summarize short transcripts directly; for longer transcripts, produce bounded evidence notes per chunk, recursively merge those notes in bounded groups, and run the unchanged final summary prompt over the merged evidence.

**Tech Stack:** Python 3.11, pytest, Ollama HTTP API (`voicenotes/ollama.py`), no new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-27-voicenotes-local-design.md` (existing system design; this fix preserves its summary template, retry model, local-only boundary, and 8K memory constraint).

## Investigation Evidence

- The captured 2026-09-04 failure had 790 timestamped raw segments and approximately 8,731 estimated tokens, but its clean transcript had zero timestamped segments and only three paragraphs.
- 366 of 790 raw segments matched the investigated Chinese filler-only definition. This is useful noise-reduction evidence, but it does not justify altering the stored raw ASR artifact or treating every acknowledgment as hallucinated speech.
- Applying the conservative collapse rule in this plan leaves 547 paragraphs and approximately 7,018 estimated tokens. The unchanged summary prompt raises that to approximately 7,472 input tokens before reserving any output, so cleanup chunking alone does not complete the real failing session within an 8K context.
- `process_session()` currently sends the entire raw transcript to one cleanup call and `_valid_text()` accepts any non-empty cleanup response, so a summary-like response becomes a valid permanent artifact.
- The former plan's literal Task 3 would raise `UnboundLocalError`: it added a `clean_transcript()` function while `process_session()` already assigned a local string named `clean_transcript` later in the same scope.
- The former oversized-paragraph test contradicted the stated bound by explicitly allowing a paragraph larger than the chunk budget.
- Ollama non-streaming responses include `done`, `done_reason`, `prompt_eval_count`, and `eval_count`. The current wrapper discards all completion metadata and returns only `response`.
- Ollama documents that larger contexts require more memory. The existing design intentionally chose `num_ctx=8192` for the supported 16GB baseline, so this plan does not double it without a measured hardware result.

## Scope Decisions

- This plan fixes both LLM stages that scale with meeting length. Cleanup is chunked without semantic reduction; summary uses a bounded map/reduce path only when the cleaned transcript does not fit one direct summary chunk.
- Cleanup supports any number of ordinary timestamped Whisper paragraphs. A single paragraph that exceeds the per-request budget fails explicitly instead of being split mid-segment or silently overflowing.
- Preserve the existing `SUMMARY_PROMPT` text and final Markdown schema. Intermediate evidence-note prompts are internal and never written as the final artifact.
- Filler collapsing is applied only to the derived cleanup input. `transcript_raw.md` remains verbatim and is never rewritten merely to remove filler.
- Only runs of at least three consecutive occurrences of the same normalized filler token, separated by no more than two seconds, are collapsed. Empty segments, mixed fillers, isolated/two-item runs, and widely separated acknowledgments remain unchanged.
- Keep `OLLAMA_CONTEXT_LENGTH = 8192`. Chunking is the load-bearing fix; a larger context requires separate 16GB memory/latency evidence.

## Global Constraints

- Do not modify `voicenotes/state.py`'s `validate_summary()` or the existing `SUMMARY_PROMPT` text/template.
- Preserve the raw Whisper segment text and timestamps in `transcript_raw.md` exactly as `write_raw_transcript()` writes them.
- Preserve retry idempotency: valid upstream artifacts remain reusable, and replacing an upstream artifact invalidates only its generated downstream artifacts.
- Do not add dependencies. Token estimation remains a local heuristic with large context headroom; Ollama completion metadata and timestamp coverage provide the fail-closed backstop.
- All new logic must be tested before it is wired into `process_session`; run the full suite (`~/.voicenotes/venv/bin/python -m pytest -q`) after every task.
- Run all commands from the isolated implementation worktree's repository root. Do not implement on `main`, and do not push or publish as part of these tasks.
- Follow existing code style in `voicenotes/pipeline.py`, `voicenotes/ollama.py`, and their tests: double-quoted strings, existing import grouping, and reuse of `config()`, `paths()`, and `summary_body()` test helpers.

---

### Task 1: Make Ollama generation fail closed

**Files:**
- Modify: `voicenotes/ollama.py:45-61`
- Test: `tests/test_ollama.py`

**Interfaces:**
- Preserves: `OLLAMA_CONTEXT_LENGTH = 8192`.
- Changes: `generate(model: str, prompt: str, timeout_seconds: int = 1800, max_output_tokens: int | None = None) -> str`.
- Guarantees: `generate()` raises `RuntimeError` for an incomplete response, `done_reason == "length"`, or a blank response; `max_output_tokens` maps to Ollama's `options.num_predict` only when provided.

- [ ] **Step 1: Update the payload test and add failing completion tests**

In `tests/test_ollama.py`, replace `test_generate_posts_non_streaming_payload` with:

```python
def test_generate_posts_non_streaming_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse({"response": "clean transcript", "done": True, "done_reason": "stop"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = generate("qwen2.5:14b", "Prompt text", timeout_seconds=1800, max_output_tokens=2048)

    assert response == "clean transcript"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["payload"] == {
        "model": "qwen2.5:14b",
        "prompt": "Prompt text",
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 2048},
        "keep_alive": "30s",
    }
    assert captured["timeout"] == 1800
```

Add immediately after it:

```python
def test_generate_rejects_incomplete_response(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeResponse({"response": "partial", "done": False}),
    )

    with pytest.raises(RuntimeError, match="did not complete"):
        generate("qwen2.5:14b", "Prompt text")


def test_generate_rejects_length_stop(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeResponse({"response": "partial", "done": True, "done_reason": "length"}),
    )

    with pytest.raises(RuntimeError, match="token limit"):
        generate("qwen2.5:14b", "Prompt text")


def test_generate_rejects_blank_response(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeResponse({"response": "   ", "done": True, "done_reason": "stop"}),
    )

    with pytest.raises(RuntimeError, match="blank response"):
        generate("qwen2.5:14b", "Prompt text")
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_ollama.py -k "generate" -v
```

Expected: the payload test fails because `generate()` does not accept `max_output_tokens`; the three completion tests fail because response metadata is ignored.

- [ ] **Step 3: Implement completion checks and the optional output cap**

Replace `generate()` in `voicenotes/ollama.py` with:

```python
def generate(
    model: str,
    prompt: str,
    timeout_seconds: int = OLLAMA_TIMEOUT_SECONDS,
    max_output_tokens: int | None = None,
) -> str:
    options: dict[str, object] = {
        "temperature": OLLAMA_TEMPERATURE,
        "num_ctx": OLLAMA_CONTEXT_LENGTH,
    }
    if max_output_tokens is not None:
        options["num_predict"] = max_output_tokens
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
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
    if body.get("done") is not True:
        raise RuntimeError("Ollama generation did not complete")
    if body.get("done_reason") == "length":
        raise RuntimeError("Ollama generation stopped at its token limit")
    generated = str(body.get("response", ""))
    if not generated.strip():
        raise RuntimeError("Ollama returned a blank response")
    return generated
```

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_ollama.py -v
~/.voicenotes/venv/bin/python -m pytest -q
```

Expected: 7 Ollama tests pass; the full suite passes.

- [ ] **Step 5: Commit**

```bash
git add voicenotes/ollama.py tests/test_ollama.py
git commit -m "Fail closed on incomplete Ollama generations"
```

---

### Task 2: Add bounded paragraph chunking helpers

**Files:**
- Modify: `voicenotes/pipeline.py` (imports and constants/helpers before `CLEANUP_PROMPT`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `estimate_tokens(text: str) -> int`.
- Produces: `chunk_paragraphs(paragraphs: list[str], max_tokens: int) -> list[list[str]]`.
- Produces: `CLEANUP_CHUNK_MAX_TOKENS = 3000` and `CLEANUP_MAX_OUTPUT_TOKENS = 3500`.
- Guarantees: separators count toward the budget, and an individually oversized paragraph raises instead of bypassing the bound.

- [ ] **Step 1: Add failing helper tests**

Add `chunk_paragraphs` and `estimate_tokens` to the import from `voicenotes.pipeline`, then add:

```python
def test_estimate_tokens_counts_cjk_chars_as_one_token_each():
    assert estimate_tokens("你好") == 2


def test_estimate_tokens_counts_other_chars_at_four_per_token():
    assert estimate_tokens("hello") == 2


def test_estimate_tokens_combines_cjk_and_other_counts():
    assert estimate_tokens("你好hello") == 4


def test_chunk_paragraphs_counts_separator_within_budget():
    paragraphs = ["A" * 8, "B" * 8]

    assert chunk_paragraphs(paragraphs, max_tokens=5) == [["A" * 8, "B" * 8]]


def test_chunk_paragraphs_splits_when_separator_exceeds_budget():
    paragraphs = ["A" * 8, "B" * 8]

    assert chunk_paragraphs(paragraphs, max_tokens=4) == [["A" * 8], ["B" * 8]]


def test_chunk_paragraphs_rejects_oversized_single_paragraph():
    with pytest.raises(ValueError, match="paragraph exceeds chunk budget"):
        chunk_paragraphs(["X" * 200], max_tokens=25)


def test_chunk_paragraphs_empty_input_returns_empty_list():
    assert chunk_paragraphs([], max_tokens=25) == []
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "estimate_tokens or chunk_paragraphs" -v
```

Expected: collection fails because the helpers do not exist.

- [ ] **Step 3: Implement the helpers**

Add `import math` after `import json`, then add after `TRANSCRIPT_MIN_CHARACTERS`:

```python
CLEANUP_CHUNK_MAX_TOKENS = 3000
CLEANUP_MAX_OUTPUT_TOKENS = 3500


def estimate_tokens(text: str) -> int:
    cjk_chars = sum(1 for ch in text if "一" <= ch <= "鿿")
    other_chars = len(text) - cjk_chars
    return cjk_chars + math.ceil(other_chars / 4)


def chunk_paragraphs(paragraphs: list[str], max_tokens: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    separator_tokens = estimate_tokens("\n\n")
    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        if paragraph_tokens > max_tokens:
            raise ValueError(f"transcript paragraph exceeds chunk budget: {paragraph_tokens} > {max_tokens}")
        added_tokens = paragraph_tokens + (separator_tokens if current else 0)
        if current and current_tokens + added_tokens > max_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
            added_tokens = paragraph_tokens
        current.append(paragraph)
        current_tokens += added_tokens
    if current:
        chunks.append(current)
    return chunks
```

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "estimate_tokens or chunk_paragraphs" -v
~/.voicenotes/venv/bin/python -m pytest -q
```

Expected: all 7 new helper tests pass; the full suite passes.

- [ ] **Step 5: Commit**

```bash
git add voicenotes/pipeline.py tests/test_pipeline.py
git commit -m "Add fail-closed transcript chunking helpers"
```

---

### Task 3: Collapse only safe filler runs in derived cleanup input

**Files:**
- Modify: `voicenotes/pipeline.py` (add transcript-parsing and filler helpers near `write_raw_transcript`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `collapse_filler_runs(paragraphs: list[str]) -> list[str]`.
- Preserves: original paragraph strings unless at least three consecutive timestamped paragraphs contain the same normalized filler token and each gap is at most two seconds.
- Does not consume or mutate the segment dictionaries passed to `write_raw_transcript()`.

- [ ] **Step 1: Add failing filler-collapse tests**

Add `collapse_filler_runs` to the import from `voicenotes.pipeline`, then add:

```python
def test_collapse_filler_runs_merges_three_close_identical_fillers():
    paragraphs = [
        "[00:00:01 - 00:00:02] 嗯。",
        "[00:00:02 - 00:00:03] 嗯",
        "[00:00:03 - 00:00:04] 嗯",
    ]

    assert collapse_filler_runs(paragraphs) == ["[00:00:01 - 00:00:04] 嗯。"]


def test_collapse_filler_runs_preserves_two_fillers():
    paragraphs = [
        "[00:00:01 - 00:00:02] 嗯",
        "[00:00:02 - 00:00:03] 嗯",
    ]

    assert collapse_filler_runs(paragraphs) == paragraphs


def test_collapse_filler_runs_preserves_mixed_fillers():
    paragraphs = [
        "[00:00:01 - 00:00:02] 嗯",
        "[00:00:02 - 00:00:03] 呃",
        "[00:00:03 - 00:00:04] 啊",
    ]

    assert collapse_filler_runs(paragraphs) == paragraphs


def test_collapse_filler_runs_preserves_widely_separated_fillers():
    paragraphs = [
        "[00:00:01 - 00:00:02] 嗯",
        "[00:00:10 - 00:00:11] 嗯",
        "[00:00:20 - 00:00:21] 嗯",
    ]

    assert collapse_filler_runs(paragraphs) == paragraphs


def test_collapse_filler_runs_does_not_invent_text_for_empty_segments():
    paragraphs = [
        "[00:00:01 - 00:00:02] ",
        "[00:00:02 - 00:00:03] ",
        "[00:00:03 - 00:00:04] ",
    ]

    assert collapse_filler_runs(paragraphs) == paragraphs


def test_collapse_filler_runs_preserves_non_filler_paragraphs():
    paragraphs = ["[00:00:01 - 00:00:02] meaningful content"]

    assert collapse_filler_runs(paragraphs) == paragraphs


def test_collapse_filler_runs_handles_empty_list():
    assert collapse_filler_runs([]) == []
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "collapse_filler_runs" -v
```

Expected: collection fails because `collapse_filler_runs` does not exist.

- [ ] **Step 3: Implement timestamp parsing and conservative collapse**

Add `import re` after `from pathlib import Path`, then add near `write_raw_transcript`:

```python
FILLER_TOKENS = {"嗯", "呃", "啊", "嗯嗯", "呃呃"}
FILLER_RUN_MIN_SEGMENTS = 3
FILLER_MAX_GAP_SECONDS = 2
TRANSCRIPT_PARAGRAPH = re.compile(
    r"^\[(?P<start>\d{2}:\d{2}:\d{2}) - (?P<end>\d{2}:\d{2}:\d{2})\](?: (?P<text>.*))?$"
)


def _timestamp_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _parse_timestamped_paragraph(paragraph: str) -> tuple[str, str, str] | None:
    match = TRANSCRIPT_PARAGRAPH.fullmatch(paragraph)
    if match is None:
        return None
    return match.group("start"), match.group("end"), match.group("text") or ""


def _normalized_filler(text: str) -> str | None:
    stripped = text.strip().strip("。.!?！？,，")
    return stripped if stripped in FILLER_TOKENS else None


def collapse_filler_runs(paragraphs: list[str]) -> list[str]:
    collapsed: list[str] = []
    run: list[tuple[str, str, str, str]] = []
    run_token: str | None = None

    def flush_run() -> None:
        nonlocal run_token
        if len(run) < FILLER_RUN_MIN_SEGMENTS:
            collapsed.extend(item[0] for item in run)
        elif run:
            collapsed.append(f"[{run[0][1]} - {run[-1][2]}] {run[0][3].strip()}")
        run.clear()
        run_token = None

    for paragraph in paragraphs:
        parsed = _parse_timestamped_paragraph(paragraph)
        token = _normalized_filler(parsed[2]) if parsed is not None else None
        if parsed is None or token is None:
            flush_run()
            collapsed.append(paragraph)
            continue
        start, end, text = parsed
        gap = _timestamp_seconds(start) - _timestamp_seconds(run[-1][2]) if run else 0
        if run and (token != run_token or gap > FILLER_MAX_GAP_SECONDS):
            flush_run()
        if not run:
            run_token = token
        run.append((paragraph, start, end, text))
    flush_run()
    return collapsed
```

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "collapse_filler_runs" -v
~/.voicenotes/venv/bin/python -m pytest -q
```

Expected: all 7 filler tests pass; the full suite passes. No production call site changes in this task.

- [ ] **Step 5: Commit**

```bash
git add voicenotes/pipeline.py tests/test_pipeline.py
git commit -m "Add conservative cleanup-only filler collapse"
```

---

### Task 4: Add validated chunked cleanup and wire it into the pipeline

**Files:**
- Modify: `voicenotes/pipeline.py` (`PROMPT_VERSION`, cleanup helpers, and `process_session`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `collapse_filler_runs()`, `chunk_paragraphs()`, `CLEANUP_CHUNK_MAX_TOKENS`, and Task 1's `ollama.generate(..., max_output_tokens=...)`.
- Produces: `clean_transcript(model: str, raw_text: str) -> str`.
- Guarantees: every source timestamp appears exactly once and in order in each cleaned chunk before the final clean artifact is written.

- [ ] **Step 1: Add failing cleanup tests**

Add `clean_transcript` to the import from `voicenotes.pipeline`, then add:

```python
def test_clean_transcript_makes_one_bounded_call_for_small_input(monkeypatch):
    prompts = []

    def fake_generate(model, prompt, timeout_seconds=1800, max_output_tokens=None):
        prompts.append((prompt, max_output_tokens))
        return "[00:00:00 - 00:00:01] cleaned transcript"

    monkeypatch.setattr("voicenotes.ollama.generate", fake_generate)

    result = clean_transcript("qwen2.5:14b", "[00:00:00 - 00:00:01] raw transcript")

    assert result == "[00:00:00 - 00:00:01] cleaned transcript"
    assert len(prompts) == 1
    assert prompts[0][1] == 3500


def test_clean_transcript_splits_and_concatenates_in_order(monkeypatch):
    monkeypatch.setattr("voicenotes.pipeline.CLEANUP_CHUNK_MAX_TOKENS", 15)
    prompts = []
    responses = iter(
        [
            "[00:00:00 - 00:00:01] " + "A" * 20,
            "[00:00:01 - 00:00:02] " + "B" * 20,
        ]
    )

    def fake_generate(model, prompt, timeout_seconds=1800, max_output_tokens=None):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr("voicenotes.ollama.generate", fake_generate)
    raw = "\n\n".join(
        [
            "[00:00:00 - 00:00:01] " + "A" * 20,
            "[00:00:01 - 00:00:02] " + "B" * 20,
        ]
    )

    result = clean_transcript("qwen2.5:14b", raw)

    assert len(prompts) == 2
    assert result == raw


def test_clean_transcript_preserves_45_minutes_of_timestamped_segments(monkeypatch):
    paragraphs = [
        f"[{format_timestamp(index * 3)} - {format_timestamp((index + 1) * 3)}] marker-{index:03d}"
        for index in range(900)
    ]
    raw = "\n\n".join(paragraphs)
    prompts = []

    def echo_cleanup(model, prompt, timeout_seconds=1800, max_output_tokens=None):
        prompts.append((prompt, max_output_tokens))
        return prompt.rsplit("Transcript:\n", 1)[1].strip()

    monkeypatch.setattr("voicenotes.ollama.generate", echo_cleanup)

    cleaned = clean_transcript("qwen2.5:14b", raw)

    assert len(prompts) > 1
    assert cleaned == raw
    assert cleaned.count("[00:") == 900
    assert all(max_output_tokens == 3500 for _, max_output_tokens in prompts)
    assert all(estimate_tokens(prompt) + max_output_tokens < 8192 for prompt, max_output_tokens in prompts)


def test_clean_transcript_rejects_missing_or_reordered_timestamps(monkeypatch):
    raw = "\n\n".join(
        [
            "[00:00:00 - 00:00:01] first",
            "[00:00:01 - 00:00:02] second",
        ]
    )
    responses = iter(
        [
            "[00:00:00 - 00:00:01] first",
            "[00:00:01 - 00:00:02] second\n\n[00:00:00 - 00:00:01] first",
        ]
    )
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: next(responses))

    for _ in range(2):
        with pytest.raises(RuntimeError, match="timestamps missing or reordered"):
            clean_transcript("qwen2.5:14b", raw)


def test_clean_transcript_rejects_missing_segment_content(monkeypatch):
    monkeypatch.setattr(
        "voicenotes.ollama.generate",
        lambda *args, **kwargs: "[00:00:00 - 00:00:01] ",
    )

    with pytest.raises(RuntimeError, match="segment content missing"):
        clean_transcript("qwen2.5:14b", "[00:00:00 - 00:00:01] raw transcript")


def test_process_session_uses_multiple_cleanup_chunks(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    monkeypatch.setattr("voicenotes.pipeline.CLEANUP_CHUNK_MAX_TOKENS", 15)
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr(
        "voicenotes.pipeline.transcribe_audio",
        lambda audio, models: [
            {"start": 0.0, "end": 1.0, "text": "A" * 20},
            {"start": 1.0, "end": 2.0, "text": "B" * 20},
        ],
    )
    responses = iter(
        [
            "[00:00:00 - 00:00:01] " + "A" * 20,
            "[00:00:01 - 00:00:02] " + "B" * 20,
            summary_body("fresh"),
        ]
    )
    prompts = []

    def fake_generate(model, prompt, timeout_seconds=1800, max_output_tokens=None):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr("voicenotes.ollama.generate", fake_generate)

    process_session(session, config(tmp_path), paths(tmp_path))

    assert len(prompts) == 3
    assert (session / "transcript_clean.md").read_text(encoding="utf-8").count("[00:00:") == 2


def test_process_session_preserves_raw_fillers_but_collapses_cleanup_input(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr(
        "voicenotes.pipeline.transcribe_audio",
        lambda audio, models: [
            {"start": 0.0, "end": 1.0, "text": "real content"},
            {"start": 1.0, "end": 2.0, "text": "嗯"},
            {"start": 2.0, "end": 3.0, "text": "嗯"},
            {"start": 3.0, "end": 4.0, "text": "嗯"},
        ],
    )
    responses = iter(
        [
            "[00:00:00 - 00:00:01] real content\n\n[00:00:01 - 00:00:04] 嗯",
            summary_body("fresh"),
        ]
    )
    prompts = []

    def fake_generate(model, prompt, timeout_seconds=1800, max_output_tokens=None):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr("voicenotes.ollama.generate", fake_generate)

    process_session(session, config(tmp_path), paths(tmp_path))

    raw = (session / "transcript_raw.md").read_text(encoding="utf-8")
    assert raw.count("嗯") == 3
    assert "[00:00:01 - 00:00:04] 嗯" in prompts[0]
```

Also update the existing strict fake in `test_process_session_writes_all_artifacts` to accept the new optional keyword:

```python
monkeypatch.setattr(
    "voicenotes.ollama.generate",
    lambda model, prompt, timeout_seconds=1800, max_output_tokens=None: next(responses),
)
```

Update `test_invalid_summary_is_saved_raw_and_fails` so cleanup succeeds before the deliberately invalid summary response:

```python
responses = iter(
    [
        "[00:00:00 - 00:00:01] hello",
        "Here is the summary\n\n## Meeting Metadata",
    ]
)
monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: next(responses))
```

In `test_retry_regenerates_downstream_artifacts_after_invalid_raw_transcript`, change the first mocked response and its assertion from bare `fresh clean` to the timestamp-preserving form:

```python
responses = iter(
    [
        "[00:00:00 - 00:00:01] fresh clean",
        summary_body("fresh"),
    ]
)

# Existing assertion in that test:
assert (session / "transcript_clean.md").read_text(encoding="utf-8") == "[00:00:00 - 00:00:01] fresh clean\n"
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "clean_transcript or multiple_cleanup_chunks or preserves_raw_fillers" -v
```

Expected: collection fails because `clean_transcript` does not exist.

- [ ] **Step 3: Implement timestamp coverage validation and chunked cleanup**

Add after `collapse_filler_runs`:

```python
def _validate_cleaned_chunk(source: str, cleaned: str) -> None:
    source_paragraphs = source.strip().split("\n\n")
    cleaned_paragraphs = cleaned.strip().split("\n\n")
    source_parts = [_parse_timestamped_paragraph(paragraph) for paragraph in source_paragraphs]
    cleaned_parts = [_parse_timestamped_paragraph(paragraph) for paragraph in cleaned_paragraphs]
    source_timestamps = [(part[0], part[1]) for part in source_parts if part is not None]
    cleaned_timestamps = [(part[0], part[1]) for part in cleaned_parts if part is not None]
    if len(source_timestamps) != len(source_paragraphs) or source_timestamps != cleaned_timestamps:
        raise RuntimeError("clean transcript validation failed: timestamps missing or reordered")
    if len(cleaned_parts) != len(source_parts) or any(part is None for part in cleaned_parts):
        raise RuntimeError("clean transcript validation failed: timestamps missing or reordered")
    for source_part, cleaned_part in zip(source_parts, cleaned_parts):
        if source_part is not None and cleaned_part is not None and source_part[2].strip() and not cleaned_part[2].strip():
            raise RuntimeError("clean transcript validation failed: segment content missing")


def clean_transcript(model: str, raw_text: str) -> str:
    if not raw_text.strip():
        raise RuntimeError("raw transcript validation failed")
    paragraphs = collapse_filler_runs(raw_text.strip().split("\n\n"))
    chunks = chunk_paragraphs(paragraphs, CLEANUP_CHUNK_MAX_TOKENS)
    cleaned_chunks: list[str] = []
    for chunk in chunks:
        source = "\n\n".join(chunk)
        cleaned = ollama.generate(
            model,
            CLEANUP_PROMPT.format(transcript_raw=source),
            max_output_tokens=CLEANUP_MAX_OUTPUT_TOKENS,
        ).strip()
        _validate_cleaned_chunk(source, cleaned)
        cleaned_chunks.append(cleaned)
    return "\n\n".join(cleaned_chunks)
```

- [ ] **Step 4: Wire cleanup into `process_session` without shadowing the function**

Replace the existing cleanup branch with:

```python
        clean_path = session / "transcript_clean.md"
        if needs_clean:
            raw_transcript = raw_path.read_text(encoding="utf-8")
            cleaned_output = clean_transcript(config.ollama_model, raw_transcript)
            atomic_write_text(clean_path, cleaned_output + "\n")
            if not _valid_text(clean_path):
                raise RuntimeError("clean transcript validation failed")
```

In the summary branch, rename the existing local variable so it cannot shadow the helper:

```python
        summary_path = session / "summary.md"
        if needs_summary:
            cleaned_transcript_text = clean_path.read_text(encoding="utf-8")
            generated = ollama.generate(config.ollama_model, SUMMARY_PROMPT.format(transcript_clean=cleaned_transcript_text))
```

Change:

```python
PROMPT_VERSION = "2026-09-03-v3"
```

to:

```python
PROMPT_VERSION = "2026-09-06-v4"
```

Update the existing prompt-version assertion to the same exact value.

- [ ] **Step 5: Run focused and full tests**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -v
~/.voicenotes/venv/bin/python -m pytest -q
```

Expected: all pipeline tests, including the 7 new cleanup tests, pass; the full suite passes.

- [ ] **Step 6: Commit**

```bash
git add voicenotes/pipeline.py tests/test_pipeline.py
git commit -m "Clean transcripts in validated bounded chunks"
```

---

### Task 5: Add bounded direct and hierarchical summarization

**Files:**
- Modify: `voicenotes/pipeline.py` (summary prompts/constants, `summarize_transcript`, and summary call site)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `SUMMARY_CHUNK_MAX_TOKENS = 3000`, `SUMMARY_NOTES_MAX_OUTPUT_TOKENS = 2000`, `SUMMARY_REDUCE_INPUT_MAX_TOKENS = 4500`, and `SUMMARY_MAX_OUTPUT_TOKENS = 1536`.
- Produces: `summarize_transcript(model: str, cleaned_text: str) -> str`.
- Changes: `PROMPT_VERSION` from the cleanup-only `2026-09-06-v4` to `2026-09-06-v5` because this task adds two prompts.
- Preserves: the existing `SUMMARY_PROMPT` text, the short-transcript direct path, and final output validation.
- Guarantees: every summary-related Ollama request has bounded input and output; arbitrarily many ordinary transcript paragraphs converge through recursive evidence-note reduction.

- [ ] **Step 1: Add failing direct and hierarchical summary tests**

Add `summarize_transcript` to the import from `voicenotes.pipeline`, then add:

```python
def test_summarize_transcript_uses_direct_path_for_one_chunk(monkeypatch):
    calls = []

    def fake_generate(model, prompt, timeout_seconds=1800, max_output_tokens=None):
        calls.append((prompt, max_output_tokens))
        return summary_body("fresh")

    monkeypatch.setattr("voicenotes.ollama.generate", fake_generate)

    result = summarize_transcript("qwen2.5:14b", "[00:00:00 - 00:00:01] short")

    assert result == summary_body("fresh")
    assert len(calls) == 1
    assert "short" in calls[0][0]
    assert calls[0][1] == 1536


def test_summarize_transcript_maps_and_recursively_reduces_long_input(monkeypatch):
    monkeypatch.setattr("voicenotes.pipeline.SUMMARY_CHUNK_MAX_TOKENS", 15)
    monkeypatch.setattr("voicenotes.pipeline.SUMMARY_REDUCE_INPUT_MAX_TOKENS", 17)
    calls = []

    def fake_generate(model, prompt, timeout_seconds=1800, max_output_tokens=None):
        calls.append((prompt, max_output_tokens))
        if "Transcript chunk:\n" in prompt:
            return "M" * 32
        if "Evidence notes:\n" in prompt:
            return "R" * 32
        return summary_body("merged")

    monkeypatch.setattr("voicenotes.ollama.generate", fake_generate)
    cleaned = "\n\n".join(
        [
            "[00:00:00 - 00:00:01] " + "A" * 20,
            "[00:00:01 - 00:00:02] " + "B" * 20,
            "[00:00:02 - 00:00:03] " + "C" * 20,
        ]
    )

    result = summarize_transcript("qwen2.5:14b", cleaned)

    map_calls = [call for call in calls if "Transcript chunk:\n" in call[0]]
    reduce_calls = [call for call in calls if "Evidence notes:\n" in call[0]]
    final_calls = [call for call in calls if call not in map_calls and call not in reduce_calls]
    assert result == summary_body("merged")
    assert len(map_calls) == 3
    assert len(reduce_calls) == 3
    assert len(final_calls) == 1
    assert all(max_output_tokens == 2000 for _, max_output_tokens in map_calls + reduce_calls)
    assert final_calls[0][1] == 1536


def test_summarize_transcript_bounds_every_call_for_45_minutes(monkeypatch):
    paragraphs = [
        f"[{format_timestamp(index * 3)} - {format_timestamp((index + 1) * 3)}] marker-{index:03d}"
        for index in range(900)
    ]
    calls = []

    def fake_generate(model, prompt, timeout_seconds=1800, max_output_tokens=None):
        calls.append((prompt, max_output_tokens))
        if "Transcript chunk:\n" in prompt:
            return f"- mapped evidence {len(calls)}"
        if "Evidence notes:\n" in prompt:
            return "- reduced evidence"
        return summary_body("long meeting")

    monkeypatch.setattr("voicenotes.ollama.generate", fake_generate)

    summary = summarize_transcript("qwen2.5:14b", "\n\n".join(paragraphs))

    assert summary == summary_body("long meeting")
    assert len(calls) > 2
    assert all(max_output_tokens in {2000, 1536} for _, max_output_tokens in calls)
    assert all(estimate_tokens(prompt) + max_output_tokens < 8192 for prompt, max_output_tokens in calls)


def test_summarize_transcript_rejects_blank_input_before_ollama(monkeypatch):
    monkeypatch.setattr(
        "voicenotes.ollama.generate",
        lambda *args, **kwargs: pytest.fail("must reject before Ollama"),
    )

    with pytest.raises(RuntimeError, match="clean transcript validation failed"):
        summarize_transcript("qwen2.5:14b", "   ")
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "summarize_transcript" -v
```

Expected: collection fails because `summarize_transcript` does not exist.

- [ ] **Step 3: Add internal evidence-note prompts and bounds**

Add near `CLEANUP_CHUNK_MAX_TOKENS`:

```python
SUMMARY_CHUNK_MAX_TOKENS = 3000
SUMMARY_NOTES_MAX_OUTPUT_TOKENS = 2000
SUMMARY_REDUCE_INPUT_MAX_TOKENS = 4500
SUMMARY_MAX_OUTPUT_TOKENS = 1536
```

Add after `SUMMARY_PROMPT` without changing any existing `SUMMARY_PROMPT` text:

```python
SUMMARY_NOTES_PROMPT = """Create dense evidence notes from this chronological transcript chunk for a later meeting summary.
Preserve every concrete fact, metric, name, example, decision, action item, owner, deadline, criticism, blocker, and open question.
Preserve the transcript's English/Mandarin language choices. Do not invent information or produce the final summary headings.
Use concise Markdown bullets and retain relevant timestamps so later stages can distinguish events.

Transcript chunk:
{transcript_clean}
"""

SUMMARY_REDUCE_PROMPT = """Merge these chronological evidence-note groups into one dense set of evidence notes for a later meeting summary.
Preserve every unique fact, metric, name, example, decision, action item, owner, deadline, criticism, blocker, open question, and timestamp.
Deduplicate only genuinely repeated information. Preserve English/Mandarin language choices. Do not invent information or produce the final summary headings.
Use concise Markdown bullets.

Evidence notes:
{notes}
"""
```

Change:

```python
PROMPT_VERSION = "2026-09-06-v4"
```

to:

```python
PROMPT_VERSION = "2026-09-06-v5"
```

Update the existing prompt-version assertion to the same exact value.

- [ ] **Step 4: Implement bounded direct/map/reduce summarization**

Add after `clean_transcript`:

```python
def summarize_transcript(model: str, cleaned_text: str) -> str:
    if not cleaned_text.strip():
        raise RuntimeError("clean transcript validation failed")
    paragraphs = cleaned_text.strip().split("\n\n")
    chunks = chunk_paragraphs(paragraphs, SUMMARY_CHUNK_MAX_TOKENS)
    if len(chunks) == 1:
        return ollama.generate(
            model,
            SUMMARY_PROMPT.format(transcript_clean="\n\n".join(chunks[0])),
            max_output_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
        )

    notes = [
        ollama.generate(
            model,
            SUMMARY_NOTES_PROMPT.format(transcript_clean="\n\n".join(chunk)),
            max_output_tokens=SUMMARY_NOTES_MAX_OUTPUT_TOKENS,
        ).strip()
        for chunk in chunks
    ]
    while len(notes) > 1:
        note_groups = chunk_paragraphs(notes, SUMMARY_REDUCE_INPUT_MAX_TOKENS)
        if len(note_groups) >= len(notes):
            raise RuntimeError("summary evidence reduction did not make progress")
        notes = [
            ollama.generate(
                model,
                SUMMARY_REDUCE_PROMPT.format(notes="\n\n".join(group)),
                max_output_tokens=SUMMARY_NOTES_MAX_OUTPUT_TOKENS,
            ).strip()
            for group in note_groups
        ]

    return ollama.generate(
        model,
        SUMMARY_PROMPT.format(transcript_clean=notes[0]),
        max_output_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
    )
```

The progress guard is intentional: `num_predict=2000` lets two maximum-size evidence-note responses fit within the 4,500-token reduce budget in normal operation, but an unexpectedly expansionary response must fail rather than loop forever.

- [ ] **Step 5: Wire the wrapper into `process_session`**

Replace:

```python
            generated = ollama.generate(config.ollama_model, SUMMARY_PROMPT.format(transcript_clean=cleaned_transcript_text))
```

with:

```python
            generated = summarize_transcript(config.ollama_model, cleaned_transcript_text)
```

- [ ] **Step 6: Run focused and full tests**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "summarize_transcript" -v
~/.voicenotes/venv/bin/python -m pytest -q
```

Expected: all 4 new summary-path tests pass; the full suite passes.

- [ ] **Step 7: Commit**

```bash
git add voicenotes/pipeline.py tests/test_pipeline.py
git commit -m "Summarize long transcripts in bounded stages"
```

---

### Task 6: Preserve retry dependency freshness and add explicit clean regeneration

**Files:**
- Modify: `voicenotes/pipeline.py` (`process_session` invalidation and `retry_session` signature)
- Modify: `voicenotes/cli.py` (`retry --from-clean`)
- Test: `tests/test_pipeline.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Changes: `retry_session(session: Path, config: AppConfig, paths: Paths, from_clean: bool = False) -> None`.
- Adds CLI: `voicenotes retry SESSION --from-clean`.
- Guarantees: once raw is regenerated, stale clean/summary artifacts cannot be accepted; once clean is regenerated, a stale summary cannot be accepted.

- [ ] **Step 1: Add failing retry tests**

Add to `tests/test_pipeline.py`:

```python
def test_retry_from_clean_regenerates_derived_artifacts(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    raw = "[00:00:00 - 00:00:01] preserved raw\n"
    (session / "transcript_raw.md").write_text(raw, encoding="utf-8")
    (session / "transcript_clean.md").write_text("[00:00:00 - 00:00:01] stale clean\n", encoding="utf-8")
    (session / "summary.md").write_text(summary_body("stale"), encoding="utf-8")
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.pipeline.transcribe_audio", lambda *args: pytest.fail("raw must be reused"))
    responses = iter(["[00:00:00 - 00:00:01] fresh clean", summary_body("fresh")])
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: next(responses))

    retry_session(session, config(tmp_path), paths(tmp_path), from_clean=True)

    assert (session / "transcript_raw.md").read_text(encoding="utf-8") == raw
    assert "fresh clean" in (session / "transcript_clean.md").read_text(encoding="utf-8")
    assert "- fresh" in (session / "summary.md").read_text(encoding="utf-8")


def test_retry_does_not_accept_stale_downstream_after_cleanup_failure(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    (session / "transcript_raw.md").write_text("", encoding="utf-8")
    (session / "transcript_clean.md").write_text("[00:00:00 - 00:00:01] stale clean\n", encoding="utf-8")
    (session / "summary.md").write_text(summary_body("stale"), encoding="utf-8")
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr(
        "voicenotes.pipeline.transcribe_audio",
        lambda *args: [{"start": 0.0, "end": 1.0, "text": "fresh raw"}],
    )
    monkeypatch.setattr(
        "voicenotes.ollama.generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        retry_session(session, config(tmp_path), paths(tmp_path))

    assert (session / "transcript_raw.md").read_text(encoding="utf-8") == "[00:00:00 - 00:00:01] fresh raw\n"
    assert not (session / "transcript_clean.md").exists()
    assert not (session / "summary.md").exists()

    monkeypatch.setattr("voicenotes.pipeline.transcribe_audio", lambda *args: pytest.fail("valid raw must be reused"))
    responses = iter(["[00:00:00 - 00:00:01] fresh clean", summary_body("fresh")])
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: next(responses))

    retry_session(session, config(tmp_path), paths(tmp_path))

    assert "fresh clean" in (session / "transcript_clean.md").read_text(encoding="utf-8")
    assert "- fresh" in (session / "summary.md").read_text(encoding="utf-8")
```

Add to `tests/test_cli.py`:

```python
def test_retry_from_clean_passes_explicit_regeneration_flag(tmp_path, monkeypatch):
    session = tmp_path / "session"
    run = tmp_path / "run"
    (run / "queue").mkdir(parents=True)
    worker_paths = type("WorkerPaths", (), {"run": run})()
    captured = {}
    monkeypatch.setattr("voicenotes.cli.acquire_pipeline_lock", lambda app_paths: True)
    monkeypatch.setattr("voicenotes.cli.release_pipeline_lock", lambda app_paths: None)
    monkeypatch.setattr("voicenotes.cli.default_paths", lambda: worker_paths)
    monkeypatch.setattr("voicenotes.cli.load_config", lambda: object())

    def fake_retry(path, app_config, app_paths, from_clean=False):
        captured["path"] = path
        captured["from_clean"] = from_clean

    monkeypatch.setattr("voicenotes.cli.retry_session", fake_retry)

    assert main(["retry", str(session), "--from-clean"]) == 0
    assert captured == {"path": session, "from_clean": True}
```

Update `test_retry_command_returns_one_for_pipeline_failure` so its fake accepts the new keyword:

```python
monkeypatch.setattr(
    "voicenotes.cli.retry_session",
    lambda path, config, paths, from_clean=False: (_ for _ in ()).throw(RuntimeError("pipeline failed")),
)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "from_clean or stale_downstream" -v
~/.voicenotes/venv/bin/python -m pytest tests/test_cli.py -k "retry_from_clean" -v
```

Expected: pipeline test fails because `retry_session` does not accept `from_clean`; CLI parsing fails because `--from-clean` is unknown.

- [ ] **Step 3: Invalidate downstream artifacts only after successful upstream replacement**

In `process_session`, define all artifact paths together before calculating stage needs:

```python
        raw_path = session / "transcript_raw.md"
        clean_path = session / "transcript_clean.md"
        summary_path = session / "summary.md"
        needs_raw = not status["transcript_raw"]
        needs_clean = needs_raw or not status["transcript_clean"]
        needs_summary = needs_clean or not status["summary"]
```

After successfully writing and validating a regenerated raw transcript, add:

```python
            clean_path.unlink(missing_ok=True)
            summary_path.unlink(missing_ok=True)
            (session / "summary.raw.md").unlink(missing_ok=True)
```

After successfully writing and validating a regenerated clean transcript, add:

```python
            summary_path.unlink(missing_ok=True)
            (session / "summary.raw.md").unlink(missing_ok=True)
```

Remove the later duplicate assignments of `clean_path` and `summary_path`; do not change `needs_raw`/`needs_clean`/`needs_summary` formulas.

- [ ] **Step 4: Add explicit clean-stage regeneration**

Replace `retry_session` with:

```python
def retry_session(
    session: Path,
    config: AppConfig,
    paths: Paths,
    from_clean: bool = False,
) -> None:
    if from_clean:
        (session / "transcript_clean.md").unlink(missing_ok=True)
        (session / "summary.md").unlink(missing_ok=True)
        (session / "summary.raw.md").unlink(missing_ok=True)
    process_session(session, config, paths)
```

In `voicenotes/cli.py`, add after `retry_parser.add_argument("session")`:

```python
    retry_parser.add_argument("--from-clean", action="store_true")
```

Replace the retry call with:

```python
                retry_session(session, load_config(), paths, from_clean=args.from_clean)
```

- [ ] **Step 5: Run focused and full tests**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "from_clean or stale_downstream" -v
~/.voicenotes/venv/bin/python -m pytest tests/test_cli.py -k "retry" -v
~/.voicenotes/venv/bin/python -m pytest -q
```

Expected: both new pipeline retry tests and the new CLI test pass; the full suite passes.

- [ ] **Step 6: Commit**

```bash
git add voicenotes/pipeline.py voicenotes/cli.py tests/test_pipeline.py tests/test_cli.py
git commit -m "Keep retry artifacts dependency-consistent"
```

---

### Task 7: Verify long-transcript regressions and update public documentation

**Files:**
- Modify: `tests/test_docs.py`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-27-voicenotes-local-design.md`

**Interfaces:**
- Consumes: all behavior and the synthetic 45-minute cleanup/summary regressions from Tasks 1-6.
- Documents: raw-transcript preservation, conservative filler handling, bounded hierarchical summarization, and explicit retry.

- [ ] **Step 1: Re-run both long-transcript regressions before documentation changes**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "45_minutes" -v
```

Expected: both tests pass with multiple bounded calls; cleanup preserves all 900 synthetic timestamps.

- [ ] **Step 2: Add the README documentation**

After the paragraph ending "a 32GB Mac may choose a larger local model", add exactly:

```markdown
Long and noisy recordings are cleaned as bounded groups of timestamped paragraphs instead of one oversized LLM request. `transcript_raw.md` remains the verbatim Whisper output; only the derived cleanup input collapses runs of at least three identical Chinese filler-only segments that occur close together. Cleanup rejects blank, truncated, or timestamp-dropping model output rather than accepting a partial transcript.

Short cleaned transcripts are summarized directly. Longer transcripts use bounded evidence-note and merge stages before the same final summary prompt, so no individual Ollama request grows with the full meeting length. Blank or token-limit-stopped model responses fail explicitly instead of becoming completed artifacts.
```

In the Commands section, replace the retry sentence with:

```markdown
Use `voicenotes retry <session>` when transcription, cleanup, or summarization fails after some artifacts were already written. Retry skips valid existing artifacts and resumes from the first missing or invalid one. Use `voicenotes retry <session> --from-clean` to deliberately regenerate a non-empty but incorrect clean transcript and its summary while preserving `transcript_raw.md`.
```

- [ ] **Step 3: Update the design spec and documentation assertion**

In `docs/superpowers/specs/2026-08-27-voicenotes-local-design.md`:

- Change pipeline step 5 from `Summarize with Ollama` to `Summarize with bounded direct or hierarchical Ollama calls`.
- After the raw timestamp example and its existing cleanup-structure sentence, add:

```markdown
`transcript_raw.md` is never rewritten for cleanup. The pipeline builds a derived cleanup input, conservatively collapses only dense runs of the same filler token, and groups complete timestamped paragraphs into bounded requests. A cleanup response is accepted only when every input timestamp range appears exactly once and in the same order, with non-empty source segments still non-empty.
```

- After the summary prompt block, add:

```markdown
If the cleaned transcript fits one bounded chunk, the pipeline applies this prompt directly. Longer transcripts are mapped to dense chronological evidence notes in bounded calls, those notes are recursively merged in bounded groups, and the unchanged final summary prompt is applied to the merged evidence. Intermediate notes are not session artifacts.
```

- The embedded Summary Prompt block still shows the superseded five-section schema. Replace that block with the exact existing `SUMMARY_PROMPT` literal from `voicenotes/pipeline.py`; this synchronizes the design document without changing runtime prompt behavior.

- Replace the sentence beginning `Each Ollama call has a 30-minute timeout` with:

```markdown
Each Ollama call has a 30-minute timeout and requests an 8k context to keep memory use reasonable on 16GB Macs. Cleanup input is capped at 3,000 estimated tokens with `num_predict=3500`; summary map input is capped at 3,000, evidence reduction input at 4,500, evidence output at `num_predict=2000`, and final summary output at `num_predict=1536`. Non-streaming responses must report `done=true`; blank responses and `done_reason=length` fail explicitly. The pipeline opens the Ollama app and waits for `localhost:11434` if the local API is not already running, then verifies the configured model is present via `/api/tags` before starting model-dependent work. Missing models fail with `ollama pull` instructions.
```

- Replace the first paragraph under Retry And Validation with:

```markdown
`voicenotes retry <session>` resumes from the first missing or invalid artifact and skips valid existing artifacts. `voicenotes retry <session> --from-clean` deliberately removes the existing clean transcript and summary before resuming while preserving the raw transcript. Whenever the pipeline successfully replaces raw, it invalidates clean and summary; whenever it successfully replaces clean, it invalidates summary.
```

- Replace acceptance criteria 7, 8, and 12 with:

```markdown
7. Cleanup uses bounded timestamp-complete requests, preserves language choice, and does not summarize or translate.
8. Short transcripts use direct summarization; long transcripts use bounded hierarchical evidence reduction; final summary output has exactly the required sections in order.
12. `retry <session>` resumes from the first invalid or missing artifact, and `retry <session> --from-clean` explicitly regenerates clean and summary while preserving raw.
```

In the existing string list in `test_readme_covers_required_user_topics`, add:

```python
        "--from-clean",
        "bounded evidence-note",
```

- [ ] **Step 4: Run documentation, regression, smoke, and full checks**

Run:

```bash
~/.voicenotes/venv/bin/python -m pytest tests/test_docs.py -v
~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "45_minutes" -v
./scripts/smoke-test.sh
~/.voicenotes/venv/bin/python -m pytest -q
```

Expected: documentation tests pass, the synthetic long-transcript regressions use multiple bounded calls and preserve all 900 cleanup timestamps, the existing local smoke test passes, and the full suite reports 106 passing tests (75 baseline + 31 new tests).

- [ ] **Step 5: Run the private semantic acceptance fixture from a disposable copy**

This gate runs only on the development machine that has the original ignored session. Set the source path privately; do not write it into the repository:

```bash
test -n "${VOICENOTES_REGRESSION_SESSION:-}"
regression_root="$(mktemp -d)"
mkdir "$regression_root/session"
cp "$VOICENOTES_REGRESSION_SESSION/audio.wav" "$regression_root/session/audio.wav"
cp "$VOICENOTES_REGRESSION_SESSION/transcript_raw.md" "$regression_root/session/transcript_raw.md"
cp "$VOICENOTES_REGRESSION_SESSION/transcript_clean.md" "$regression_root/session/transcript_clean.md"
cp "$VOICENOTES_REGRESSION_SESSION/summary.md" "$regression_root/session/summary.md"
~/.voicenotes/venv/bin/python -m voicenotes retry "$regression_root/session" --from-clean
cmp "$VOICENOTES_REGRESSION_SESSION/transcript_raw.md" "$regression_root/session/transcript_raw.md"
cleaned_timestamp_count="$(rg -c '^\[[0-9]{2}:[0-9]{2}:[0-9]{2} - [0-9]{2}:[0-9]{2}:[0-9]{2}\]' "$regression_root/session/transcript_clean.md")"
test "$cleaned_timestamp_count" -eq 547
~/.voicenotes/venv/bin/python -c 'from pathlib import Path; from voicenotes.state import validate_summary; import sys; ok, reason = validate_summary(Path(sys.argv[1])); print(reason); raise SystemExit(0 if ok else 1)' "$regression_root/session/summary.md"
printf '%s\n' "$regression_root/session"
```

Expected: retry succeeds; `cmp` confirms the raw transcript is byte-identical; the cleaned transcript contains 547 derived timestamped paragraphs; `validate_summary()` prints `ok`. Manually re-score the same private 24-claim rubric used during investigation and require at least 20/24 claims in `summary.md`. If the semantic threshold fails, do not publish: retain the disposable fixture for prompt diagnosis and revise Task 5's evidence prompts/budgets with a new failing regression that contains only synthetic content.

- [ ] **Step 6: Confirm the public diff contains no private meeting content**

Run:

```bash
git diff --check
git diff -- README.md docs/superpowers/specs/2026-08-27-voicenotes-local-design.md tests/test_pipeline.py tests/test_docs.py
git ls-files | rg '(^|/)VoiceNotes/|transcript_(raw|clean)\.md$|summary(\.raw)?\.md$'
```

Expected: `git diff --check` exits zero; manual diff review finds only synthetic markers and public behavior documentation; the tracked-artifact search returns no matches.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/superpowers/specs/2026-08-27-voicenotes-local-design.md tests/test_docs.py
git commit -m "Document bounded long-transcript processing"
```

## Final Verification Gate

Before declaring the implementation ready for publication:

```bash
~/.voicenotes/venv/bin/python -m pytest -q
git status --short
git log --oneline --decorate -8
```

Expected: 106 tests pass; only intentional implementation commits are present; the worktree is clean. Then use `superpowers:requesting-code-review` for a whole-branch review and `superpowers:finishing-a-development-branch` to present integration options. Pushing or publishing remains an explicit user-authorized action outside this plan.
