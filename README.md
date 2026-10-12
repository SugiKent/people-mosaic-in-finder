# People Mosaic

Mac の中だけで動画の顔を検出し、白い雲や自分の透過 PNG を顔に追従させるツールです。Finder のクイックアクションから起動しても、必ず人物選択とプレビューの画面が開きます。

## この Mac で使う

Finder で動画を右クリックし、**クイックアクション → People Mosaic** を選びます。単独で開く場合は ZIP 版の `~/Applications/People Mosaic.app`、clone 版の `build/People Mosaic.app` を開いてください。

1. **「動画を選ぶ…」**、または画面へのドロップで動画を開きます。顔を探し、最初は全員の顔に雲を付けます。
2. 左側の動画を見ながら、**外したい雲をクリック**します。人物を識別できた雲は、その人物候補の動画全体に反映します。小さい顔や看板・絵など、人物を識別できない検出もクリックで外せます。この場合は連続して追跡できる範囲だけに反映し、カット・追跡の途切れ・別の対象との混同があると再び隠します。**「外した雲を1つ戻す」**でクリックを取り消せます。
3. 右側の **「雲を外す人」** では、顔カードの「雲を外す」にチェックして選べます。写真のクリックは登場場面への移動です。人数が多いときは **「人物一覧を広げる」** を使います。登場順・よく映る順で並べ替え、「選んだ人だけ」で見直せます。選択と並べ替えは拡大一覧でも共有します。**「すべての顔を隠す」**は人物選択・クリック除外・検出の細かさを元に戻します。
4. 右側の **「調整・設定」** で、映像を見ながら調整できます。**「顔を見つける細かさ」**を右へ動かすと誤検出を減らし、左へ動かすと小さい顔・横顔も多く拾います。標準は 0.80、範囲は 0.30〜0.99。変更はすぐプレビューと保存に反映し、再解析しません。高くすると実際の顔も隠れなくなることがあります。「雲の見た目」から透過 PNG を選ぶこともできます。
5. 検出漏れを補うときは「動画で範囲を囲む」を押してドラッグします。追加した場所は動画全体の同じ位置で隠れます。「範囲追加を終了」で通常の操作へ戻ります。手動の範囲は雲のクリックでは外さず、「追加した範囲を戻す」で取り消します。設定の「小さい顔も探す」へ変更する方法もありますが、動画を再解析し、人物選択と追加範囲もリセットします。
6. 確認できたら **「動画を保存する」** を押します。標準は共有用の MP4。画質を保つ MKV は「調整・設定」で選べます。保存前の確認チェックはありません。保存・処理中止のボタンは画面下に固定し、設定が多くてもスクロールして操作できます。

画面は Mac の外観設定に関係なく **ライトモード**で表示します。人物選択と設定を切り替えても、映像は常に表示します。動画は確認用に間引いて再生し、プレビューの音声は再生しません。重なった雲はクリック位置の最前面が対象です。

出力は入力動画と同じフォルダに `{元のファイル名（拡張子なし）}_mosaic.mp4` または `_mosaic.mkv` で自動保存します。例：`20260927_080143.mp4` → `20260927_080143_mosaic.mp4`。同名ファイルがある場合は `_mosaic_1.mp4`、`_mosaic_2.mp4`…と、空いている最初の番号に保存します。既存ファイルは上書きしません。

雲の細かな揺れを抑えるため、顔サイズの 15%（6〜24px）までの位置変化には追従を控えます。雲の拡大・縮小も抑え、十分小さい状態が 0.6 秒続いてからゆっくり縮めます。大きな移動には追従を速め、検出した顔全体を覆う大きさを保ちます。この更新を反映するには、アプリを開き直して動画を再解析してください。

人物の自動判定は完全ではありません。同じ人が複数候補に分かれることがあります。表示を許可した候補でも、判定が曖昧なフレームは隠します。検出された顔だけを自動加工するため、公開前に書き出した動画の全体を確認してください。

M1 Max の 8 個の高性能コアを想定し、4 本の解析処理（各 2 スレッド）を使用します。解析用のデコードは Apple の VideoToolbox を優先し、初期化できない場合は 2 スレッドのソフトウェア処理に切り替えます。全フレームを解析し、待機フレームは最大 8 枚に制限します。追跡は長辺 960px の画像で全顔を一括処理します。高速モードの縮小は顔検出用だけで、人物識別は元の画像から切り出し、書き出す動画の解像度は維持します。

書き出しは 4 本の合成処理を並列に動かし、待機は最大 8 イベントに制限します。雲画像の前処理と、同じ大きさ・透過・色条件の合成係数を再利用します（処理ごとのキャッシュは最大 64 MB）。MP4 は Mac の VideoToolbox で H.264 / HEVC を圧縮し、H.264 の初期化ができない環境ではソフトウェア圧縮へ切り替えます。変換が必要な音声も、映像を再エンコードせず AAC に変換します。MKV は元の画素を維持するソフトウェアデコードと、並列 FFV1 圧縮を使います。

## 画質と形式

| 出力 | 用途 | 映像 | 音声 |
| --- | --- | --- | --- |
| MP4 | 公開・通常の再生 | SDR は H.264、HLG HDR は HEVC 10bit。高画質の再圧縮 | AAC / ALAC / MP3 はコピー。それ以外は AAC に変換 |
| MKV | 保存・追加加工 | FFV1 可逆圧縮。PNG が触れる範囲以外は、元のデコード済み YUV サンプルを維持 | コピー |

ハードウェア MP4 は元動画のビットレートを下回らない高めのビットレートで圧縮します。ソフトウェア H.264 の CRF 16 と同じ画質・容量にはなりません。入力よりファイルが大きくなることがあります。

MP4 で「劣化が一切ない」は保証しません。MKV は可逆圧縮ですが、加工した顔は元の映像と異なります。透過 PNG の縁は合成され、色差が間引かれた映像では接触する色差サンプルも変わります。MKV の時刻はコンテナの精度（通常 1ms）に丸められます。元の圧縮ビット列をそのまま保つことは、動画を加工する以上できません。

iPhone の MOV に通常の AAC と APAC 空間音声が両方ある場合は、AAC を保持して補助の APAC を除きます。APAC しかない動画は未対応として停止します。

解像度とフレーム数を維持し、可変フレームレートの各フレームの時刻を使います。動画全体を RGB に変換して戻す処理は行いません。90 度単位の回転は画素の並べ替えで反映します。HLG の雲は、MP4 では PNG の白を最大白へ合わせてグレー寄りになる減光を避け、MKV では従来の HDR 基準白に合わせます。HDR 情報を維持して出力します。画面の HDR プレビューは SDR に変換した確認用表示です。

対応: SDR YUV 420/422/444 の 8/10bit、BT.2020 YUV 420 10bit の HLG HDR、正方形ピクセル、偶数サイズ、プログレッシブ映像。PQ HDR / Dolby Vision、インターレース、任意角度の回転は未対応です。

MKV は QuickTime Player で再生できない場合があります。HDR MP4 の表示には HEVC / HLG 対応のプレイヤーと公開先が必要です。

## PNG の条件

自分で選ぶ PNG は 20 MB 以下、縦横 32〜4096 ピクセル。中央の縦横 50% を十分な濃さの画像で覆ってください。読み込み時に中央を不透明にし、周辺の透過は維持します。顔の縦横より十分大きく配置し、不透明な中央で顔を覆います。同梱の雲画像は、指定された PNG の色・透過・元の寸法をそのまま読み込みます。中央の不透明化は行いません。

## ZIP からインストール

macOS 15 以降の Apple Silicon の Mac 向けです。Python、FFmpeg、GUI ライブラリ、顔認識モデルをアプリに同梱しています。Homebrew・git clone・手動 build は不要で、ZIP のダウンロード後はネット接続なしでインストールできます。

1. [最新の GitHub Release](https://github.com/SugiKent/people-mosaic-in-finder/releases/latest) の **`People-Mosaic-0.1.0-macOS-arm64.zip`** をダウンロードします。GitHub が自動生成する「Source code (zip)」や開発者向けの `Sources` ZIP ではなく、名前に `macOS-arm64` が入った ZIP を選んでください。
2. ZIP を展開し、**`Install.command`** を開きます。同梱のアプリをコピーし、Finder に登録します。依存ツールのインストールやモデルのダウンロードは行いません。
3. Finder で動画を右クリック → **クイックアクション → People Mosaic**。単独で開く場合は **`~/Applications/People Mosaic.app`** を開きます。

Apple の Developer ID 署名・公証は取得していません。macOS が起動を止めた場合は、システム設定の **プライバシーとセキュリティ → このまま開く** から、ダウンロードした `Install.command` または `People Mosaic.app` を許可してください。[Apple の案内](https://support.apple.com/ja-jp/102445)も参照できます。

本体は `~/Library/Application Support/People Mosaic/versions/` の版ごとのフォルダへ保存します。インストール完了後は、ダウンロードした ZIP と展開フォルダを削除できます。Finder アクションは `~/Library/Services/People Mosaic.workflow` です。既にこのツールのアクションがある場合はバックアップして新版へ切り替えます。同名の別アプリ・別アクションは上書きしません。

更新は People Mosaic を閉じてから、新しい Release の ZIP を展開して `Install.command` を開くだけです。以前の版は保持し、準備に失敗した場合は起動先を切り替えません。起動時に GitHub から自動更新することはありません。

ZIP 版を削除する場合は `~/Applications/People Mosaic.app`、`~/Library/Services/People Mosaic.workflow`、`~/Library/Application Support/People Mosaic` をゴミ箱へ移動します。

ZIP 版はビルド済みなので、起動のたびに clone したソースを読み直す仕組みではありません。更新する際は新しい ZIP をインストールしてください。ソースを編集して毎回読み込ませたい場合は、次の clone 版を使います。

### Finder に People Mosaic が出ないとき

1. `Install.command` の画面に「インストール完了」と表示されたか確認します。展開したアプリを開くだけでは Finder に登録されません。未完了なら、最新版の ZIP の `Install.command` を実行してください。
2. `~/Applications/People Mosaic.app` と `~/Library/Services/People Mosaic.workflow` があるか確認します。アプリを単独で開ける場合は、その画面から動画を選んで使えます。
3. Finder で **MP4 / MOV などの動画ファイル自体**を選び、右クリックの **「サービス」**と **「クイックアクション」**の両方を確認します。フォルダや何もない場所を右クリックしても表示されません。
4. クイックアクションの「カスタマイズ」、またはシステム設定の「キーボード → キーボードショートカット → サービス」で People Mosaic が有効か確認します。Finder 拡張の設定場所は macOS のバージョンによって異なります。[Apple の案内](https://support.apple.com/ja-jp/guide/automator/aut73234890a/mac)も参照してください。

ターミナルでインストール先と設定ファイルを確認する場合は、次を実行します。ファイルや設定は変更しません。

```sh
ls -ld "$HOME/Applications/People Mosaic.app" "$HOME/Library/Services/People Mosaic.workflow"
plutil -lint "$HOME/Library/Services/People Mosaic.workflow/Contents/Info.plist"
plutil -lint "$HOME/Library/Services/People Mosaic.workflow/Contents/document.wflow"
```

インストール済みなのに反映されない場合、応急処置としてターミナルから次を実行し、メニューを開き直せます。これは macOS のサービス一覧を再検出する診断用ツールです。Apple は将来の提供を保証していないため、通常のインストール処理では公開 API `NSUpdateDynamicServices()` を使います。

```sh
/System/Library/CoreServices/pbs -update
```

## clone してインストール・開発

この手順は macOS 15 以降の Apple Silicon の Mac（M1 Max / 64GB を想定）向けです。ターミナルを使用します。Homebrew が未導入なら、先に [Homebrew 公式サイト](https://brew.sh/ja/) の手順でインストールしてください。

### 1. コードと依存ツールを取得

```sh
git clone https://github.com/SugiKent/people-mosaic-in-finder.git
cd people-mosaic-in-finder
brew install uv ffmpeg python@3.12
```

このフォルダは、使用中に移動・削除しない場所に置いてください。

### 2. アプリを準備して Finder に登録

プロジェクトのフォルダ内で実行します。

```sh
./scripts/setup.command
./scripts/install_finder.command
```

Python.org の古い Python.app を経由しないよう、Homebrew の Apple Silicon 版 Python 3.12 を優先します。初回はライブラリと OpenCV 公式モデル（約 39 MB）を取得します。モデルは SHA-256 で検証します。セットアップ後の顔検出・プレビュー・書き出しではネットワーク通信をしません。

生成物は `build/People Mosaic.app` と `build/People Mosaic.workflow` です。Finder アクションは `~/Library/Services/People Mosaic.workflow` に登録されます。

`setup.command` が起動用の app / workflow も生成するので、別途 build を実行する必要はありません。アプリ本体の Python ソースはパッケージにコピーせず、このフォルダの `src/` を起動ごとに読み込みます。clone だけでは依存関係や顔モデルが揃わないため、初回のセットアップと Finder 登録は必要です。

### 3. 起動を確認

Finder で動画を右クリックし、**クイックアクション → People Mosaic** を選びます。顔の解析と人物選択の画面が開けば準備完了です。アクションが表示されない場合は、システム設定で People Mosaic のサービス / Finder 拡張機能が有効か確認してください。

単独で起動する場合は次を実行します。

```sh
open "build/People Mosaic.app"
```

「透過 PNG を選ぶ」から独自の雲画像を読み込めます。完成した動画は入力と同じフォルダに `_mosaic.mp4` または `_mosaic.mkv` を付けて保存されます。

## clone 版の更新・再インストール・削除

アプリを閉じ、プロジェクトのフォルダ内で更新します。

```sh
git pull --ff-only
```

ソースコードだけの更新なら、アプリを開き直すだけで反映されます。フォルダの場所が同じなら再ビルド・Finder への再登録は不要です。Finder から起動すると、その時点のローカルソースを読み込みます。起動時に GitHub から自動取得する仕組みではないため、GitHub の更新を取り込むには `git pull` が必要です。

依存関係・モデル・起動用スクリプトを更新した場合は、続けて準備をやり直します。

```sh
./scripts/setup.command
```

アプリと Finder アクションはこのプロジェクトの絶対パスを参照します。フォルダを移動した場合や、別の場所のコードで登録し直す場合は、Finder の「移動 → フォルダへ移動」で `~/Library/Services` を開き、既存の `People Mosaic.workflow` を別の場所へ退避してから、移動先で次を実行します。

```sh
./scripts/setup.command
./scripts/install_finder.command
```

インストーラは同名の既存アクションを上書きしません。「同名のアクションが既にあります」と表示された場合も、上記の退避後に再実行してください。Finder アクションを削除するには `~/Library/Services/People Mosaic.workflow` をゴミ箱へ移動します。

## 開発と検証

```sh
uv sync --frozen --python /opt/homebrew/bin/python3.12
.venv/bin/ruff check src scripts tests
.venv/bin/pytest -q
./scripts/launch.command /absolute/path/to/video.mp4
```

検証記録は [validation.md](https://github.com/SugiKent/people-mosaic-in-finder/blob/main/docs/quality/validation.md)、画質・顔を表示する判定・プライバシーの契約は [contracts.md](https://github.com/SugiKent/people-mosaic-in-finder/blob/main/docs/domain/video-masking/contracts.md) を参照してください。

## Release の作成

main への push（PR のマージを含む）で GitHub Actions がテスト・ZIP の作成・Release の公開まで実行します。最新版の main と一致するビルドだけ公開し、実行番号とコミットを含む固有タグを使います。既存の公開 ZIP は上書きしません。手動実行も main から可能です。

ローカルで同梱アプリをビルドする場合は、Apple Silicon の Mac と Xcode Command Line Tools、uv、pkg-config が必要です。配布先の Mac にはこれらは不要です。必要なソースをコミットしてから実行します。

```sh
.venv/bin/python scripts/build_ffmpeg.py
.venv/bin/python scripts/prepare_bundle_runtime.py
build/freeze-venv/bin/python scripts/collect_bundle_licenses.py
build/freeze-venv/bin/python scripts/build_bundle.py \
  --ffmpeg build/ffmpeg/bin/ffmpeg --ffprobe build/ffmpeg/bin/ffprobe
.venv/bin/python scripts/package_release.py --ref HEAD
.venv/bin/python scripts/verify_bundle.py \
  --zip build/releases/People-Mosaic-0.1.0-macOS-arm64.zip
```

`build/releases/` にアプリ ZIP、対応するソース ZIP、SHA256 ファイルが生成されます。アプリ ZIP は指定したコミットからビルドした Python・Qt・OpenCV・PyAV、動画ツール、検証済みの公式顔モデルを含みます。個人の動画・生成画像・開発用仮想環境は含めません。Finder アクションの起動先だけをインストール先に合わせて生成します。

配布用の FFmpeg は LGPL 構成でビルドし、PyAV もそのライブラリに接続します。OpenCV は動画 I/O を外してビルドします。MP4 は Apple のハードウェアエンコーダーを優先し、SDR のソフトウェア代替は OpenH264 を使います。同梱ライブラリのライセンス・著作権表示はアプリ内の `Contents/Resources/licenses/`、対応する上流ソースとビルド手順は同じ Release の `People-Mosaic-Sources-*.zip` に収録します。ライブラリを変更してデバッグすることを制限しません。変更したアプリは `codesign --force --deep --sign - "People Mosaic.app"` でローカル署名を付け直せます。

顔検出と識別は [OpenCV の YuNet / SFace](https://docs.opencv.org/4.12.0/d0/dd4/tutorial_dnn_face.html)、保存用の可逆圧縮は [FFV1](https://github.com/FFmpeg/FFV1) を使用しています。HLG グラフィックスの基準白は [ITU-R BT.2408](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2408-8-2024-PDF-E.pdf) を参照しています。
