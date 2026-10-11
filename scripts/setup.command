#!/bin/zsh
set -eu
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null || ! command -v ffmpeg >/dev/null; then
  print '先に brew install uv ffmpeg を実行してください。'
  exit 1
fi
if [[ -x /opt/homebrew/bin/python3.12 ]]; then
  uv sync --frozen --python /opt/homebrew/bin/python3.12
else
  uv sync --frozen --python 3.12
fi
.venv/bin/python -m people_mosaic.models
.venv/bin/python scripts/build_macos.py
print '準備完了。build/People Mosaic.app を開いてください。'
print 'Finder のアクションを有効にするには scripts/install_finder.command を実行してください。'
