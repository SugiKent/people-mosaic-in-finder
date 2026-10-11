#!/bin/zsh
set -eu
package_root="${0:A:h}"
if [[ "$(/usr/bin/uname -s)" != Darwin || "$(/usr/bin/uname -m)" != arm64 ]]; then
  print 'Apple Silicon の Mac で、Rosetta を使わないターミナルから実行してください。'
  exit 1
fi
"$package_root/People Mosaic.app/Contents/MacOS/PeopleMosaic" --install "$package_root"
