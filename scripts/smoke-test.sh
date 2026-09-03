#!/usr/bin/env bash
set -euo pipefail

SESSION="$(voicenotes record-test --duration 10 | tail -n 1)"
voicenotes process "$SESSION"

for file in audio.wav transcript_raw.md transcript_clean.md summary.md; do
  test -s "$SESSION/$file"
done

for heading in \
  "## Summary" \
  "## Discussion by topic" \
  "## Feedback & critique" \
  "## Decisions" \
  "## Action items" \
  "## Blockers & open questions" \
  "## Next steps"; do
  grep -Fxq "$heading" "$SESSION/summary.md"
done

echo "Smoke test passed: $SESSION"
