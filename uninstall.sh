#!/usr/bin/env bash
set -euo pipefail

rm -f "$HOME/.hammerspoon/voicenotes.lua"

if [[ -f "$HOME/.hammerspoon/init.lua" ]]; then
  tmp="$(mktemp)"
  grep -Fxv 'require("voicenotes")' "$HOME/.hammerspoon/init.lua" > "$tmp" || true
  mv "$tmp" "$HOME/.hammerspoon/init.lua"
fi

rm -f /opt/homebrew/bin/voicenotes
rm -f /usr/local/bin/voicenotes
rm -f "$HOME/.local/bin/voicenotes"
rm -rf "$HOME/.voicenotes/run"
rm -rf "$HOME/.voicenotes/app"
rm -rf "$HOME/.voicenotes/venv"

echo "Uninstalled VoiceNotes Local app files."
echo "Did not remove Homebrew dependencies, Ollama models, or $HOME/VoiceNotes."
echo "To remove the default Ollama model and reclaim roughly 9GB, run: ollama rm qwen2.5:14b"
