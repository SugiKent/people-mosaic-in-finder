#!/bin/zsh
set -eu
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")/.."
.venv/bin/python scripts/build_macos.py
target="$HOME/Library/Services/People Mosaic.workflow"
if [[ -e "$target" ]]; then
  print '同名のアクションが既にあります。既存のものを確認してから再実行してください。'
  exit 1
fi
mkdir -p "$HOME/Library/Services"
/usr/bin/ditto 'build/People Mosaic.workflow' "$target"
/System/Library/CoreServices/pbs -flush
print 'Finder で動画を右クリック → クイックアクション → People Mosaic'
print '表示されない場合は、システム設定のサービス／Finder 拡張機能で有効にしてください。'
