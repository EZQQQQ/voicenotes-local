#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$HOME/.voicenotes/app"
VENV_DIR="$HOME/.voicenotes/venv"
CONFIG_DIR="$HOME/.voicenotes"
CONFIG_FILE="$CONFIG_DIR/config.toml"
REPO_URL="${VOICENOTES_REPO_URL:-https://github.com/ezqqqq/voicenotes-local.git}"
BREW_PREFIX="/opt/homebrew"
BREW="$BREW_PREFIX/bin/brew"
WRAPPER_DIR="$BREW_PREFIX/bin"
WRAPPER="$WRAPPER_DIR/voicenotes"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "VoiceNotes Local v1 requires Apple Silicon (arm64)." >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

"$BREW" install git ffmpeg python@3.11
"$BREW" install --cask hammerspoon ollama

mkdir -p "$CONFIG_DIR" "$CONFIG_DIR/run" "$CONFIG_DIR/models"
if [[ "$REPO_URL" == /* ]]; then
  rm -rf "$APP_DIR"
  git clone "$REPO_URL" "$APP_DIR"
elif [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" pull --ff-only
else
  rm -rf "$APP_DIR"
  git clone "$REPO_URL" "$APP_DIR"
fi

"$BREW_PREFIX/bin/python3.11" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"

if [[ ! -f "$CONFIG_FILE" ]]; then
  cp "$APP_DIR/config.example.toml" "$CONFIG_FILE"
fi

if [[ ! -w "$WRAPPER_DIR" ]]; then
  WRAPPER_DIR="$HOME/.local/bin"
  WRAPPER="$WRAPPER_DIR/voicenotes"
  mkdir -p "$WRAPPER_DIR"
  echo "Add $WRAPPER_DIR to PATH to use voicenotes from the command line."
fi

cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
export PYTHONPATH="$APP_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
exec "$VENV_DIR/bin/python" -m voicenotes "\$@"
EOF
chmod +x "$WRAPPER"

if [[ "${NONINTERACTIVE:-}" != "1" ]]; then
  echo "VoiceNotes will download local models. Combined download size is approximately 12GB."
  if ! read -r -p "Continue? [y/N] " answer </dev/tty; then
    echo "No interactive terminal is available. Re-run with NONINTERACTIVE=1 to allow model downloads." >&2
    exit 1
  fi
  case "$answer" in
    y|Y|yes|YES) ;;
    *) echo "Cancelled before model downloads."; exit 1 ;;
  esac
fi

OLLAMA_MODEL="$("$WRAPPER" config --json | "$VENV_DIR/bin/python" -c 'import json, sys; print(json.load(sys.stdin)["ollama_model"])')"
ollama pull "$OLLAMA_MODEL"
"$WRAPPER" download-whisper-model

mkdir -p "$HOME/.hammerspoon"
cp "$APP_DIR/hammerspoon/voicenotes.lua" "$HOME/.hammerspoon/voicenotes.lua"
touch "$HOME/.hammerspoon/init.lua"
if ! grep -Fxq 'require("voicenotes")' "$HOME/.hammerspoon/init.lua"; then
  cp "$HOME/.hammerspoon/init.lua" "$HOME/.hammerspoon/init.lua.$(date +%Y%m%d%H%M%S).bak"
  printf '\nrequire("voicenotes")\n' >> "$HOME/.hammerspoon/init.lua"
fi

open -g hammerspoon://reload || true

echo "If Cmd+\` conflicts on this Mac, clear macOS Move focus to next window in System Settings > Keyboard > Keyboard Shortcuts > Keyboard."
echo "Grant permissions in System Settings > Privacy & Security > Microphone and Accessibility."
"$WRAPPER" doctor || true
