# yomitoku-staged-lowmem

画素から図中の活字を、箱・文字列・スコア・読み順で出す部品。分野は知らない。
店名も項目名も付けない。人物・服・部屋の説明も付けない。

## 出すもの

| 段 | 入口 | 出口 |
|---|---|---|
| 検出 | 画像 | `points.json`（四隅と検出スコア） |
| 認識 | 画像 + points | `ocr.json`（検出順の活字） |
| 組み立て | ocr.json | `document.json`（読み順と隔離） |

検出と認識は別プロセス。同じプロセスに載せない。
字の帯だけ切る。絵として何が描いてあるかは出さない。

出口の欄は [schema/ocr_raw.v0.md](schema/ocr_raw.v0.md) と [schema/document_ocr.v1.md](schema/document_ocr.v1.md)。

## 動かしたいとき

実行用 ONNX は 40MiB 未満の parts として git にある（合計 270MB 程度）。結合してから使う。

1. `pip install -r requirements.runtime.txt`
2. `python3 weights/join_weights.py`
3. `python3 weights/verify_weights.py`
4. `./run_staged.sh path/to/image.jpg results`

無い・違うと失敗する。ネットへ取りに行かない。置き方の詳細は [weights/README.md](weights/README.md)。

普通の実行は cvsurf と ONNX Runtime と numpy を使う。yomitoku / torch / pyclipper / opencv-python は import しない。

## 試験（重みなし）

```bash
python3 tests/test_assemble.py
python3 tests/test_default_no_yomitoku.py
python3 tests/test_cv2_surface.py
python3 tests/test_offset.py
python3 tests/test_join_weights.py
python3 tests/test_cvsurf_geom.py
python3 tests/test_cvsurf_io.py
```

凍結した出力から本文が残るか、余計な import が無いか、箱の膨らましが動いていないか、parts が元に戻るかを見る。

## 注意

- 結合後の ONNX / safetensors は git に載せない。parts とハッシュは `weights/manifest.json`。
- 元重みのライセンスは CC BY-NC-SA 4.0。商用は元を見る。
- JPEG / PNG / BMP の読み、縮小、輪郭、歪み補正は `cvsurf/`。ONNX Runtime は推論に残す。

## 他の紙

GitHub が最初に見せるのはこの README だけ。他の紙はここから行く。

| 紙 | 何か |
|---|---|
| [weights/README.md](weights/README.md) | parts の結合と、Drive からの別置き |
| [schema/](schema/) | 出てくる JSON の契約 |
| [SESSION.md](SESSION.md) | 続きの作業記録。来客用ではない |
| [CODING.md](CODING.md) | 直す人向けの書き方 |
| [comparison/](comparison/) | 元実装との比較。普通の実行には要らない |
