# People Mosaic

Mac の中だけで動画の顔を検出し、白い雲や自分の透過 PNG を顔に追従させるツールです。Finder のクイックアクションから起動しても、必ず人物選択と確認の画面が開きます。

## この Mac で使う

Finder で動画を右クリックし、**クイックアクション → People Mosaic** を選びます。単独で開く場合は `build/People Mosaic.app` を開いてください。

1. 動画を選びます。全フレームを解析するため、4K・60fps は時間がかかります。
2. 最初は全員を隠します。**雲を付けない人物候補**だけにチェックします。
3. 必要なら「透過 PNG を選ぶ」で画像を変更します。
4. 加工後の動画を再生・スクラブし、横顔、小さい顔、一時的な検出漏れを確認します。「隠す範囲を追加」で、動画全体の固定範囲を追加できます。
5. 確認にチェックを入れ、出力形式を選んで「確定して書き出す」を押します。設定を変えると確認チェックは解除されます。

出力は入力動画と同じフォルダに `{元のファイル名（拡張子なし）}_mosaic.mp4` または `_mosaic.mkv` で自動保存します。例：`20260927_080143.mp4` → `20260927_080143_mosaic.mp4`。同名の出力がある場合は上書きせず停止します。

人物の自動判定は完全ではありません。同じ人が複数候補に分かれることがあります。表示を許可した候補でも、判定が曖昧なフレームは隠します。検出された顔だけを自動加工するため、公開前に書き出した動画の全体を確認してください。

## 画質と形式

| 出力 | 用途 | 映像 | 音声 |
| --- | --- | --- | --- |
| MP4 | 公開・通常の再生 | SDR は H.264、HLG HDR は HEVC 10bit。高画質の再圧縮 | AAC / ALAC / MP3 はコピー。それ以外は AAC に変換 |
| MKV | 保存・追加加工 | FFV1 可逆圧縮。PNG が触れる範囲以外は、元のデコード済み YUV サンプルを維持 | コピー |

MP4 で「劣化が一切ない」は保証しません。MKV は可逆圧縮ですが、加工した顔は元の映像と異なります。透過 PNG の縁は合成され、色差が間引かれた映像では接触する色差サンプルも変わります。MKV の時刻はコンテナの精度（通常 1ms）に丸められます。元の圧縮ビット列をそのまま保つことは、動画を加工する以上できません。

解像度とフレーム数を維持し、可変フレームレートの各フレームの時刻を使います。動画全体を RGB に変換して戻す処理は行いません。90 度単位の回転は画素の並べ替えで反映します。HLG の雲は HDR の基準白に合わせ、HDR 情報を維持して出力します。画面の HDR プレビューは SDR に変換した確認用表示です。

対応: SDR YUV 420/422/444 の 8/10bit、BT.2020 YUV 420 10bit の HLG HDR、正方形ピクセル、偶数サイズ、プログレッシブ映像。PQ HDR / Dolby Vision、インターレース、任意角度の回転は未対応です。

MKV は QuickTime Player で再生できない場合があります。HDR MP4 の表示には HEVC / HLG 対応のプレイヤーと公開先が必要です。

## PNG の条件

PNG は 20 MB 以下、縦横 32〜4096 ピクセル。中央の縦横 50% は完全に不透明にしてください。周辺は透過できます。顔の縦横より十分大きく配置し、不透明な中央で顔を覆います。白い雲は最初から入っています。

## 初回セットアップ / 再構築

Apple Silicon の Mac での実行を想定しています。

```sh
brew install uv ffmpeg python@3.12
./scripts/setup.command
./scripts/install_finder.command
```

Python.org の古い Python.app を経由しないよう、Homebrew の Apple Silicon 版 Python 3.12 を優先します。初回はライブラリと OpenCV 公式モデル（約 39 MB）を取得します。モデルは SHA-256 で検証します。セットアップ後の顔検出・プレビュー・書き出しではネットワーク通信をしません。

アプリと Finder アクションはこのプロジェクトの絶対パスを参照します。フォルダを移動した場合は `scripts/build_macos.py` で再構築し、既存アクションを確認して入れ替えてください。インストーラは同名の既存アクションを上書きしません。Finder アクションの削除は `~/Library/Services/People Mosaic.workflow` をゴミ箱へ移動します。

## 開発と検証

```sh
uv sync --frozen --python /opt/homebrew/bin/python3.12
.venv/bin/ruff check src scripts tests
.venv/bin/pytest -q
./scripts/launch.command /absolute/path/to/video.mp4
```

検証記録は [docs/quality/validation.md](docs/quality/validation.md)、画質・顔を表示する判定・プライバシーの契約は [docs/domain/video-masking/contracts.md](docs/domain/video-masking/contracts.md) を参照してください。

顔検出と識別は [OpenCV の YuNet / SFace](https://docs.opencv.org/4.12.0/d0/dd4/tutorial_dnn_face.html)、保存用の可逆圧縮は [FFV1](https://github.com/FFmpeg/FFV1) を使用しています。HLG グラフィックスの基準白は [ITU-R BT.2408](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2408-8-2024-PDF-E.pdf) を参照しています。
