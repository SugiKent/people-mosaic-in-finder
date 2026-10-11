#!/bin/zsh
set -eu
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
package_root="${0:A:h}"
if [[ "$(/usr/bin/uname -s)" != Darwin || "$(/usr/bin/uname -m)" != arm64 ]]; then
  print 'Apple Silicon の Mac で、Rosetta を使わないターミナルから実行してください。'
  exit 1
fi
if [[ ! -x /opt/homebrew/bin/brew ]]; then
  print '先に https://brew.sh/ja/ の手順で Homebrew をインストールしてください。'
  print 'その後、この Install.command をもう一度開いてください。'
  exit 1
fi
typeset -a missing_tools
missing_tools=()
[[ -x /opt/homebrew/bin/uv ]] || missing_tools+=(uv)
[[ -x /opt/homebrew/bin/ffmpeg ]] || missing_tools+=(ffmpeg)
[[ -x /opt/homebrew/bin/python3.12 ]] || missing_tools+=(python@3.12)
if (( ${#missing_tools} )); then
  /opt/homebrew/bin/brew install "${missing_tools[@]}"
fi
/opt/homebrew/bin/python3.12 "$package_root/scripts/install_release.py"
print '完了しました。Finder のクイックアクション → People Mosaic から使えます。'
