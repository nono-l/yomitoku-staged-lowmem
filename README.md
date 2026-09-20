# yomitoku-staged-lowmem

画素から図中の活字を、箱・文字列・スコア・読み順で出す部品。分野は知らない。
店名も項目名も付けない。人物・服・部屋の説明も付けない。

人間と、あとから入った AI の両方が読める説明。
方針は [CODING.md](CODING.md)。次のセッションへの記録は [SESSION.md](SESSION.md)。
契約は [schema/ocr_raw.v0.md](schema/ocr_raw.v0.md) と [schema/document_ocr.v1.md](schema/document_ocr.v1.md)。

## 出すもの

| 段 | 入口 | 出口 |
|---|---|---|
| 検出 | 画像 | `points.json`（四隅と検出スコア） |
| 認識 | 画像 + points | `ocr.json`（ocr_raw/v0。検出順） |
| 組み立て | ocr.json | `document.json`（document_ocr/v1。読み順と隔離） |

検出と認識は別プロセス。同じ作業台に載せない。
字の帯だけ切る。絵として何が描いてあるかは出さない。

## 今動く面

既定経路は yomitoku / torch / pyclipper を import しない。
役は cv2 / numpy / onnxruntime。残す cv2 名は [tests/test_cv2_surface.py](tests/test_cv2_surface.py) が凍結している。

```bash
pip install -r requirements.runtime.txt
python3 weights/verify_weights.py
./run_staged.sh path/to/image.jpg results
```

`verify_weights.py` は既定で runtime の ONNX と字表を見る。safetensors は `--origin`。
実体は git に無い。置き場所は [weights/README.md](weights/README.md)。

## モデルを載せない試験

```bash
python3 tests/test_assemble.py
python3 tests/test_default_no_yomitoku.py
python3 tests/test_cv2_surface.py
python3 tests/test_offset.py
```

凍結した v0 から本文が残るか、import が漏れないか、cv2 の面が膨らまないか、offset の箱が動いていないかを見る。

torch 比較と export は [comparison/](comparison/)。そこだけ yomitoku 0.15.0 が要る。

## 注意

- 別置き ONNX / safetensors は git に載せない。ハッシュは `weights/manifest.json`。
- 元重みのライセンスは CC BY-NC-SA 4.0。商用は元を見る。
- JPEG・輪郭・warp・onnxruntime は腰。裁断なしに剥がない。
