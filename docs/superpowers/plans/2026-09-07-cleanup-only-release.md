# Cleanup-only release

## Scope and acceptance

User-approved scope: separate, install, and publish the verified cleanup repair; leave the unsuccessful summary redesign isolated. This supersedes the summary/release scope of the historical September 4 plan. The release must preserve raw transcripts, bound cleanup input, reject lossy cleanup, and support opt-in regeneration. Existing recordings are not automatically rewritten.

## Global Constraints

- Keep the baseline summary prompt, generation path, template and validator unchanged. No experimental summary windows, JSON schema, reasoning controls, or new model defaults.
- Keep Ollama context at 8192 and default model unchanged. Cleanup input budget is 1200 estimated tokens; cleanup output cap is 3500 tokens.
- Preserve raw transcript and audio bytes. Never fall back to raw text as accepted cleanup. Failed cleanup must not leave stale accepted clean/summary artifacts.
- Preserve complete derived-input paragraphs and timestamp occurrences in order during model cleanup. The explicitly requested conservative filler-run collapse happens beforehand on derived input only; raw paragraphs remain unchanged. Reject incomplete/length-limited/blank model responses. Bisect only completed but invalid cleanup, not generation errors.
- No private recordings, source facts, captures or rubrics in tracked files. Keep README additions short and retain its accuracy limitation warning.
- Use TDD for extraction; no unrelated refactors, dependencies, config changes or AI coauthor attribution. Implementer writes only within this worktree and does not push or install.

### Task 1: Extract verified cleanup and recovery safeguards

Worktree: `/Users/bytedance/.voicenotes/app/.worktrees/cleanup-only-release`.
Read-only implementation source: `/Users/bytedance/.voicenotes/app/.worktrees/cleanup-context-overflow-fix`, commit `87c7ce2`. Port only the accepted cleanup behavior, not that branch wholesale. Do not read its ignored scratch workspace or private evidence.

Files: `voicenotes/pipeline.py`, `voicenotes/ollama.py`, `voicenotes/cli.py`, their existing tests, and `README.md`. Keep the baseline summary prompt, generation path, template and validator unchanged. No experimental summary windows, JSON schema, reasoning controls, or new model defaults. Keep context 8192 and default model unchanged. Use PROMPT_VERSION `2026-09-07-cleanup-v1`.

1. First add/adapt the source branch's cleanup-only public-boundary regressions and observe expected RED failures against this baseline. Test token estimates, separators and paragraph chunking (including oversized paragraph preflight); conservative filler collapse; timestamp/order/paragraph validation; contraction rejection and bisection; blank paragraphs; generation failures; downstream artifact invalidation and retry recovery. Port no experimental summary tests. Adapt baseline mocked cleanup responses to valid timestamp-preserving text, preserving tests' actual intent.
2. Port `estimate_tokens`, `chunk_paragraphs`, the verified CLEANUP_PROMPT timestamp instructions, cleanup constants (1200 input / 3500 output), filler parser/helpers/collapse, `_validate_cleaned_chunk`, `_clean_chunk`, and `clean_transcript` from the source. Preserve exact accepted behavior: CJK-aware estimate; never split paragraphs; only collapse runs of at least 3 same recognized fillers with gaps at most 2 seconds; raw unchanged; blank timestamp paragraphs bypass model; ordered timestamp and nonempty body validation; bodies of at least 40 non-whitespace characters retain at least 80%; recursively bisect invalid completed groups and fail invalid singleton. Generation errors propagate without bisection.
3. Integrate cleanup into `process_session` without importing the summary redesign. After successful transcription, invalidate generated clean/summary/summary.raw before replacing raw. When cleanup is required, invalidate summary/summary.raw before attempting cleanup so failure cannot expose a stale summary. Preserve successful clean when subsequent legacy summary fails. Use the existing single legacy `ollama.generate` summary call and existing summary publication/validation logic.
4. Port only `max_output_tokens: int | None = None` to `ollama.generate`, setting num_predict only when supplied. Add completion, length-limit and blank-response checks. Leave default request payload unchanged. Do not port response_format/system/think parameters. Test actual HTTP payload and response boundaries with realistic done/done_reason fixtures, including rejected thinking-only responses.
5. Port `retry_session(..., from_clean: bool = False)` and CLI `retry --from-clean`; delete only derived clean/summary/summary.raw before regeneration, reusing raw/audio. Normal retry remains unchanged. Test parser, dispatch, raw preservation and failure paths.
6. README: add one short command sentence explaining `--from-clean` regenerates cleanup and summary while keeping raw; add one short sentence that cleanup uses bounded chunks and raw stays unchanged. Retain the existing warning about long/noisy recording accuracy. Do not claim summary is chunked or fixed.
7. Run focused tests during iteration, then the complete pytest suite, inspect diff for accidentally imported summary code/private data, commit the repair, and write the report with RED/GREEN commands/results, file list, commit and concerns. No subagents, installation, push, model calls or original-session writes.

## Controller verification and release

- Independently review the task diff, then conduct final whole-branch review.
- Replay previously verified private cleanup HTTP captures against this release; require byte-identical accepted cleanup. Exercise recovery through the real CLI on a disposable session with synthetic legacy-summary output; distinguish replay from new model-quality evidence.
- Re-run full tests; ensure no private evidence is tracked and README stays concise. Check installed main is clean and remote has not moved unexpectedly.
- User has authorized installation and publication: fast-forward local main to the release, run installed CLI/import checks and tests, then push main without force and verify remote SHA. Do not change model config or regenerate original notes.
- Preserve the unsuccessful experimental branch and private evidence. Report actual installed/published commit and remaining summary limitations.
