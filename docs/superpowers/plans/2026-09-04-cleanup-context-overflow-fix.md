# Cleanup Context-Overflow & ASR Filler Noise Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `transcript_clean.md` from turning into a summary on long/noisy recordings by chunking cleanup so it never overflows Ollama's context window regardless of meeting length, and by collapsing ASR filler-hallucination runs before they inflate the transcript.

**Architecture:** Two independent, additive changes to `voicenotes/pipeline.py`'s processing pipeline: (1) `clean_transcript()` splits the raw transcript into token-budgeted chunks (never splitting a paragraph) and calls `ollama.generate()` once per chunk instead of once for the whole transcript, concatenating the results; (2) `collapse_filler_runs()` merges consecutive filler-only/empty ASR segments (e.g. runs of bare "嗯") into a single representative segment before `write_raw_transcript()` ever sees them. `voicenotes/ollama.py`'s `OLLAMA_CONTEXT_LENGTH` is also raised as cheap complementary headroom for a single chunk.

**Tech Stack:** Python 3.11, pytest, Ollama HTTP API (`voicenotes/ollama.py`), no new dependencies.

**Spec:** none — this is a bounded fix (existing pipeline flow, no new subsystem). Root-cause diagnosis and design were presented and approved in chat directly: `transcript_raw.md` for a real 2026-09-04 session was ~8,731 estimated tokens (already over the prior `OLLAMA_CONTEXT_LENGTH=8192`) with 46% of its segments (365/790) being pure ASR filler/hallucination noise, causing the cleanup LLM call to fall back to summarizing instead of faithfully reproducing.

## Global Constraints

- Do not modify `voicenotes/state.py`'s `validate_summary()` or the `SUMMARY_PROMPT`/summary template — that was redesigned and shipped in a prior change; out of scope here.
- Preserve existing idempotency/retry behavior in `process_session`/`retry_session` (`artifact_status()`'s `needs_raw`/`needs_clean`/`needs_summary` gating must keep working unchanged).
- All new logic must be covered by tests before being wired into `process_session`; run the full suite (`~/.voicenotes/venv/bin/python -m pytest -q`) after each task, not just the new test file.
- Follow existing code style in `voicenotes/pipeline.py` and `tests/test_pipeline.py` exactly (double-quoted strings, existing import grouping, existing `config()`/`paths()`/`summary_body()` test helpers — reuse them, don't duplicate).

---

### Task 1: Raise Ollama context length headroom

**Files:**
- Modify: `voicenotes/ollama.py:12`
- Test: `tests/test_ollama.py:76`

**Interfaces:**
- Produces: `voicenotes.ollama.OLLAMA_CONTEXT_LENGTH = 16384` (was `8192`) — consumed by `voicenotes.ollama.generate()`'s existing `options.num_ctx` payload field (no signature change).

- [ ] **Step 1: Update the failing test**

In `tests/test_ollama.py`, in `test_generate_posts_non_streaming_payload`, change:

```python
        "options": {"temperature": 0.2, "num_ctx": 8192},
```

to:

```python
        "options": {"temperature": 0.2, "num_ctx": 16384},
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/.voicenotes/app && ~/.voicenotes/venv/bin/python -m pytest tests/test_ollama.py::test_generate_posts_non_streaming_payload -v`
Expected: FAIL — `AssertionError` on the `num_ctx` value (actual `8192` vs expected `16384`).

- [ ] **Step 3: Update the constant**

In `voicenotes/ollama.py`, change:

```python
OLLAMA_CONTEXT_LENGTH = 8192
```

to:

```python
OLLAMA_CONTEXT_LENGTH = 16384
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/.voicenotes/app && ~/.voicenotes/venv/bin/python -m pytest tests/test_ollama.py -v`
Expected: all pass (4 tests).

- [ ] **Step 5: Commit**

```bash
cd ~/.voicenotes/app
git add voicenotes/ollama.py tests/test_ollama.py
git commit -m "Raise Ollama context length headroom to 16384"
```

---

### Task 2: Token-budgeted paragraph chunking helpers

**Files:**
- Modify: `voicenotes/pipeline.py` (add near the top, after the existing constants block at line 18)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `estimate_tokens(text: str) -> int` and `chunk_paragraphs(paragraphs: list[str], max_tokens: int) -> list[list[str]]` in `voicenotes/pipeline.py` — consumed by Task 3's `clean_transcript()`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_pipeline.py`, change the import line:

```python
from voicenotes.pipeline import CLEANUP_PROMPT, PROMPT_VERSION, SUMMARY_PROMPT, artifact_status, format_timestamp, process_session, retry_session, transcribe_audio
```

to:

```python
from voicenotes.pipeline import CLEANUP_PROMPT, PROMPT_VERSION, SUMMARY_PROMPT, artifact_status, chunk_paragraphs, estimate_tokens, format_timestamp, process_session, retry_session, transcribe_audio
```

Then add these tests anywhere after the imports:

```python
def test_estimate_tokens_counts_cjk_chars_as_one_token_each():
    assert estimate_tokens("你好") == 2


def test_estimate_tokens_counts_other_chars_at_four_per_token():
    assert estimate_tokens("hello") == 2  # ceil(5 / 4)


def test_estimate_tokens_combines_cjk_and_other_counts():
    assert estimate_tokens("你好hello") == 4  # 2 + ceil(5 / 4)


def test_chunk_paragraphs_keeps_paragraphs_together_within_budget():
    paragraphs = ["A" * 8, "B" * 8]  # 2 tokens each (ceil(8/4))

    assert chunk_paragraphs(paragraphs, max_tokens=10) == [["A" * 8, "B" * 8]]


def test_chunk_paragraphs_splits_when_budget_exceeded():
    paragraphs = ["A" * 40, "B" * 40, "C" * 40]  # 10 tokens each

    assert chunk_paragraphs(paragraphs, max_tokens=25) == [
        ["A" * 40, "B" * 40],
        ["C" * 40],
    ]


def test_chunk_paragraphs_keeps_oversized_single_paragraph_as_its_own_chunk():
    paragraphs = ["X" * 200]  # 50 tokens, alone exceeds the budget

    assert chunk_paragraphs(paragraphs, max_tokens=25) == [["X" * 200]]


def test_chunk_paragraphs_empty_input_returns_empty_list():
    assert chunk_paragraphs([], max_tokens=25) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/.voicenotes/app && ~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "estimate_tokens or chunk_paragraphs" -v`
Expected: FAIL with `ImportError` / `NameError` — `estimate_tokens`/`chunk_paragraphs` not defined.

- [ ] **Step 3: Implement the helpers**

In `voicenotes/pipeline.py`, add `import math` to the top imports (after `import json`, alphabetically before `from pathlib import Path`):

```python
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
```

Then add after the existing constants block (after `TRANSCRIPT_MIN_CHARACTERS = 1` and before `CLEANUP_PROMPT`):

```python
CLEANUP_CHUNK_MAX_TOKENS = 3000


def estimate_tokens(text: str) -> int:
    cjk_chars = sum(1 for ch in text if "一" <= ch <= "鿿")
    other_chars = len(text) - cjk_chars
    return cjk_chars + math.ceil(other_chars / 4)


def chunk_paragraphs(paragraphs: list[str], max_tokens: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        tokens = estimate_tokens(paragraph)
        if current and current_tokens + tokens > max_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(paragraph)
        current_tokens += tokens
    if current:
        chunks.append(current)
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/.voicenotes/app && ~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "estimate_tokens or chunk_paragraphs" -v`
Expected: all 6 pass.

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `cd ~/.voicenotes/app && ~/.voicenotes/venv/bin/python -m pytest -q`
Expected: all pass (no existing test imports/uses these names yet, so no regressions possible).

- [ ] **Step 6: Commit**

```bash
cd ~/.voicenotes/app
git add voicenotes/pipeline.py tests/test_pipeline.py
git commit -m "Add token-budgeted paragraph chunking helpers"
```

---

### Task 3: Chunked cleanup generation, wired into process_session

**Files:**
- Modify: `voicenotes/pipeline.py:16` (PROMPT_VERSION), `voicenotes/pipeline.py`'s `process_session` (the `needs_clean` branch), and add `clean_transcript()`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `estimate_tokens`, `chunk_paragraphs`, `CLEANUP_CHUNK_MAX_TOKENS` from Task 2; `ollama.generate(model: str, prompt: str, timeout_seconds: int = 1800) -> str` (existing, unchanged signature).
- Produces: `clean_transcript(model: str, raw_text: str) -> str` in `voicenotes/pipeline.py` — consumed by `process_session`'s `needs_clean` branch.

- [ ] **Step 1: Write the failing tests**

In `tests/test_pipeline.py`, change the import line:

```python
from voicenotes.pipeline import CLEANUP_PROMPT, PROMPT_VERSION, SUMMARY_PROMPT, artifact_status, chunk_paragraphs, estimate_tokens, format_timestamp, process_session, retry_session, transcribe_audio
```

to:

```python
from voicenotes.pipeline import CLEANUP_PROMPT, PROMPT_VERSION, SUMMARY_PROMPT, artifact_status, chunk_paragraphs, clean_transcript, estimate_tokens, format_timestamp, process_session, retry_session, transcribe_audio
```

Then add these tests anywhere after the imports:

```python
def test_clean_transcript_makes_one_call_for_a_small_transcript(monkeypatch):
    prompts = []
    monkeypatch.setattr(
        "voicenotes.ollama.generate",
        lambda model, prompt, timeout_seconds=1800: prompts.append(prompt) or "cleaned transcript",
    )

    result = clean_transcript("qwen2.5:14b", "[00:00:00 - 00:00:01] short transcript")

    assert len(prompts) == 1
    assert "short transcript" in prompts[0]
    assert result == "cleaned transcript"


def test_clean_transcript_splits_into_multiple_chunks_and_concatenates(monkeypatch):
    monkeypatch.setattr("voicenotes.pipeline.CLEANUP_CHUNK_MAX_TOKENS", 5)
    prompts = []
    responses = iter(["cleaned A", "cleaned B"])
    monkeypatch.setattr(
        "voicenotes.ollama.generate",
        lambda model, prompt, timeout_seconds=1800: (prompts.append(prompt), next(responses))[1],
    )

    raw_text = ("A" * 20) + "\n\n" + ("B" * 20)  # 5 tokens each; budget of 5 forces separate chunks
    result = clean_transcript("qwen2.5:14b", raw_text)

    assert len(prompts) == 2
    assert "A" * 20 in prompts[0]
    assert "B" * 20 in prompts[1]
    assert result == "cleaned A\n\ncleaned B"


def test_process_session_cleans_raw_transcript_in_multiple_chunks(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    monkeypatch.setattr("voicenotes.pipeline.CLEANUP_CHUNK_MAX_TOKENS", 5)
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr(
        "voicenotes.pipeline.transcribe_audio",
        lambda audio, models: [
            {"start": 0.0, "end": 1.0, "text": "A" * 20},
            {"start": 1.0, "end": 2.0, "text": "B" * 20},
        ],
    )
    prompts = []
    responses = iter(["cleaned A", "cleaned B", summary_body("fresh")])
    monkeypatch.setattr(
        "voicenotes.ollama.generate",
        lambda model, prompt, timeout_seconds=1800: (prompts.append(prompt), next(responses))[1],
    )

    process_session(session, config(tmp_path), paths(tmp_path))

    assert len(prompts) == 3
    assert "A" * 20 in prompts[0]
    assert "B" * 20 in prompts[1]
    clean = (session / "transcript_clean.md").read_text(encoding="utf-8")
    assert clean.strip() == "cleaned A\n\ncleaned B"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/.voicenotes/app && ~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "clean_transcript or cleans_raw_transcript_in_multiple" -v`
Expected: FAIL — `clean_transcript` not defined, and `process_session` still makes exactly one cleanup call (so the multi-chunk process_session test fails on `len(prompts) == 3`, getting 2 instead).

- [ ] **Step 3: Implement `clean_transcript` and wire it into `process_session`**

In `voicenotes/pipeline.py`, add after `write_raw_transcript` (after line 129, before `_valid_text`):

```python
def clean_transcript(model: str, raw_text: str) -> str:
    paragraphs = raw_text.strip().split("\n\n")
    chunks = chunk_paragraphs(paragraphs, CLEANUP_CHUNK_MAX_TOKENS)
    cleaned_chunks = [
        ollama.generate(model, CLEANUP_PROMPT.format(transcript_raw="\n\n".join(chunk))).strip()
        for chunk in chunks
    ]
    return "\n\n".join(cleaned_chunks)
```

Then in `process_session`, replace the `needs_clean` branch:

```python
        clean_path = session / "transcript_clean.md"
        if needs_clean:
            raw_transcript = raw_path.read_text(encoding="utf-8")
            atomic_write_text(clean_path, ollama.generate(config.ollama_model, CLEANUP_PROMPT.format(transcript_raw=raw_transcript)) + "\n")
            if not _valid_text(clean_path):
                raise RuntimeError("clean transcript validation failed")
```

with:

```python
        clean_path = session / "transcript_clean.md"
        if needs_clean:
            raw_transcript = raw_path.read_text(encoding="utf-8")
            atomic_write_text(clean_path, clean_transcript(config.ollama_model, raw_transcript) + "\n")
            if not _valid_text(clean_path):
                raise RuntimeError("clean transcript validation failed")
```

Then bump the prompt version. In `voicenotes/pipeline.py`, change:

```python
PROMPT_VERSION = "2026-09-03-v3"
```

to:

```python
PROMPT_VERSION = "2026-09-04-v4"
```

In `tests/test_pipeline.py`, update `test_prompts_preserve_raw_language_choice`:

```python
    assert PROMPT_VERSION == "2026-09-03-v3"
```

to:

```python
    assert PROMPT_VERSION == "2026-09-04-v4"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/.voicenotes/app && ~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: all pass, including the 3 new tests and the updated `PROMPT_VERSION` assertion.

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `cd ~/.voicenotes/app && ~/.voicenotes/venv/bin/python -m pytest -q`
Expected: all pass — every other existing fixture's raw transcript is short enough (well under the real 3000-token `CLEANUP_CHUNK_MAX_TOKENS` default) to still produce exactly one cleanup call, so the existing 2-item `responses = iter([...])` patterns in `test_process_session_writes_all_artifacts`, `test_retry_skips_valid_existing_artifacts`, `test_retry_regenerates_malformed_utf8_transcript`, and `test_retry_regenerates_downstream_artifacts_after_invalid_raw_transcript` are unaffected.

- [ ] **Step 6: Commit**

```bash
cd ~/.voicenotes/app
git add voicenotes/pipeline.py tests/test_pipeline.py
git commit -m "Chunk cleanup transcript generation so it scales with meeting length"
```

---

### Task 4: Collapse consecutive ASR filler/hallucination segments

**Files:**
- Modify: `voicenotes/pipeline.py` (add near `write_raw_transcript`; wire into `process_session`'s `needs_raw` branch)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing from other tasks (operates on the `list[dict[str, object]]` segment shape already used by `transcribe_audio`/`write_raw_transcript`).
- Produces: `collapse_filler_runs(segments: list[dict[str, object]]) -> list[dict[str, object]]` in `voicenotes/pipeline.py` — consumed by `process_session`'s `needs_raw` branch, applied to `transcribe_audio()`'s return value before `write_raw_transcript()`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_pipeline.py`, change the import line:

```python
from voicenotes.pipeline import CLEANUP_PROMPT, PROMPT_VERSION, SUMMARY_PROMPT, artifact_status, chunk_paragraphs, clean_transcript, estimate_tokens, format_timestamp, process_session, retry_session, transcribe_audio
```

to:

```python
from voicenotes.pipeline import CLEANUP_PROMPT, PROMPT_VERSION, SUMMARY_PROMPT, artifact_status, chunk_paragraphs, clean_transcript, collapse_filler_runs, estimate_tokens, format_timestamp, process_session, retry_session, transcribe_audio
```

Then add these tests anywhere after the imports:

```python
def test_collapse_filler_runs_merges_consecutive_filler_segments():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "real content"},
        {"start": 1.0, "end": 2.0, "text": "嗯"},
        {"start": 2.0, "end": 3.0, "text": "嗯。"},
        {"start": 3.0, "end": 4.0, "text": "嗯"},
        {"start": 4.0, "end": 5.0, "text": "more real content"},
    ]

    assert collapse_filler_runs(segments) == [
        {"start": 0.0, "end": 1.0, "text": "real content"},
        {"start": 1.0, "end": 4.0, "text": "嗯"},
        {"start": 4.0, "end": 5.0, "text": "more real content"},
    ]


def test_collapse_filler_runs_preserves_isolated_filler_segment():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "real content"},
        {"start": 1.0, "end": 2.0, "text": "嗯"},
        {"start": 2.0, "end": 3.0, "text": "more real content"},
    ]

    assert collapse_filler_runs(segments) == segments


def test_collapse_filler_runs_leaves_non_filler_transcript_untouched():
    segments = [{"start": 0.0, "end": 1.0, "text": "nothing filler here"}]

    assert collapse_filler_runs(segments) == segments


def test_collapse_filler_runs_handles_empty_segment_list():
    assert collapse_filler_runs([]) == []


def test_process_session_collapses_filler_runs_before_writing_raw_transcript(tmp_path, monkeypatch):
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
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: summary_body("fresh"))

    process_session(session, config(tmp_path), paths(tmp_path))

    raw = (session / "transcript_raw.md").read_text(encoding="utf-8")
    assert raw.count("嗯") == 1
    assert "[00:00:01 - 00:00:04] 嗯" in raw
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/.voicenotes/app && ~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -k "collapse_filler_runs or collapses_filler_runs_before" -v`
Expected: FAIL — `collapse_filler_runs` not defined, and the wiring test's `transcript_raw.md` still contains 3 separate "嗯" occurrences (not collapsed).

- [ ] **Step 3: Implement `collapse_filler_runs` and wire it into `process_session`**

In `voicenotes/pipeline.py`, add after `write_raw_transcript` and before `clean_transcript` (or right after it — either position is fine, keep both transcript-shaping helpers adjacent):

```python
FILLER_TOKENS = {"嗯", "呃", "啊", "嗯嗯", "呃呃"}


def _is_filler_segment(text: str) -> bool:
    stripped = text.strip().strip("。.!?！？,，")
    return stripped in FILLER_TOKENS or stripped == ""


def collapse_filler_runs(segments: list[dict[str, object]]) -> list[dict[str, object]]:
    collapsed: list[dict[str, object]] = []
    run: list[dict[str, object]] = []

    def flush_run() -> None:
        if not run:
            return
        if len(run) == 1:
            collapsed.append(run[0])
        else:
            collapsed.append({"start": run[0]["start"], "end": run[-1]["end"], "text": "嗯"})
        run.clear()

    for segment in segments:
        if _is_filler_segment(str(segment["text"])):
            run.append(segment)
        else:
            flush_run()
            collapsed.append(segment)
    flush_run()
    return collapsed
```

Then in `process_session`, change:

```python
        if needs_raw:
            write_raw_transcript(session, transcribe_audio(session / "audio.wav", paths))
            if not _valid_text(raw_path):
                raise RuntimeError("raw transcript validation failed")
```

to:

```python
        if needs_raw:
            segments = collapse_filler_runs(transcribe_audio(session / "audio.wav", paths))
            write_raw_transcript(session, segments)
            if not _valid_text(raw_path):
                raise RuntimeError("raw transcript validation failed")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/.voicenotes/app && ~/.voicenotes/venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: all pass, including the 5 new tests.

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `cd ~/.voicenotes/app && ~/.voicenotes/venv/bin/python -m pytest -q`
Expected: all pass — existing fixtures' segments (e.g. `"We discussed roadmap and 中文部分."`, `"hello"`) are not filler, so `collapse_filler_runs` is a no-op on them.

- [ ] **Step 6: Commit**

```bash
cd ~/.voicenotes/app
git add voicenotes/pipeline.py tests/test_pipeline.py
git commit -m "Collapse consecutive ASR filler/hallucination segments before cleanup"
```

---

### Task 5: End-to-end regression check and README note

**Files:**
- Modify: `README.md` (document the chunking/filler-collapse behavior briefly, near the existing "Why Whisper Model Is Fixed And Ollama Model Is Configurable" section)
- Test: none new — this task verifies the whole suite and does a manual smoke check

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: nothing new — closes out the plan.

- [ ] **Step 1: Run the full test suite one more time**

Run: `cd ~/.voicenotes/app && ~/.voicenotes/venv/bin/python -m pytest -q`
Expected: all tests pass (should be roughly 89: 75 original + 6 chunking + 3 clean_transcript/process_session + 5 collapse_filler_runs).

- [ ] **Step 2: Add a README note**

In `README.md`, after the existing paragraph ending "...a 32GB Mac may choose a larger local model." (end of the "Why Whisper Model Is Fixed And Ollama Model Is Configurable" section), add:

```markdown
Long or noisy recordings are handled in two ways: the raw transcript is cleaned in token-budgeted chunks rather than in one LLM call, so cleanup can't silently overflow the model's context window on a 30-45+ minute meeting; and consecutive filler-only ASR segments (e.g. runs of "嗯") are collapsed before cleanup to cut down on transcription noise.
```

- [ ] **Step 3: Verify the README doc test still passes**

Run: `cd ~/.voicenotes/app && ~/.voicenotes/venv/bin/python -m pytest tests/test_docs.py -v`
Expected: pass (this addition doesn't remove any of the required substrings `test_readme_covers_required_user_topics` checks for).

- [ ] **Step 4: Commit**

```bash
cd ~/.voicenotes/app
git add README.md
git commit -m "Document chunked cleanup and filler-collapse behavior in README"
```

- [ ] **Step 5: Push**

```bash
cd ~/.voicenotes/app
git push origin main
```
