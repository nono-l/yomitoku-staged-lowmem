# yomitoku-staged-lowmem

画素から図中の活字を、箱・文字列・スコア・読み順で出す部品。分野は知らない。

人間と、あとから入った AI の両方が読める説明。
方針は [CODING.md](CODING.md)。次のセッションへの記録は [SESSION.md](SESSION.md)。
契約は [schema/ocr_raw.v0.md](schema/ocr_raw.v0.md) と [schema/document_ocr.v1.md](schema/document_ocr.v1.md)。

## 今動く面

検出と認識を別プロセスで回し、重み無しで読み順に戻す。
既定経路は yomitoku を import しない。

```bash
pip install -r requirements.runtime.txt
./run_staged.sh path/to/image.jpg results
python3 tests/test_assemble.py
python3 tests/test_default_no_yomitoku.py
```

`python3 tests/test_assemble.py` はモデルを載せない。凍結した v0 から本文が残るかだけを見る。

torch 比較と export は [comparison/](comparison/)。そこだけ yomitoku 0.15.0 が要る。

## 注意

- 別置き ONNX / safetensors は git に載せない。ハッシュは `weights/manifest.json`。
- 元重みのライセンスは CC BY-NC-SA 4.0。商用は元を見る。
