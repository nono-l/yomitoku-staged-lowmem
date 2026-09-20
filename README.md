# yomitoku-staged-lowmem

画素から図中の活字を、箱・文字列・スコア・読み順で出す部品。分野は知らない。

人間と、あとから入った AI の両方が読める説明。
方針は [CODING.md](CODING.md)。次のセッションへの記録は [SESSION.md](SESSION.md)。
契約は [schema/ocr_raw.v0.md](schema/ocr_raw.v0.md) と [schema/document_ocr.v1.md](schema/document_ocr.v1.md)。

## 今動く面

[YomiToku](https://github.com/kotaro-kinoshita/yomitoku) の検出と認識を別プロセスで回し、重み無しで読み順に戻す。
作業台に検出器と認識器を同時に広げない。

```bash
pip install yomitoku==0.15.0
./run_staged.sh path/to/image.jpg results
python3 tests/test_assemble.py
```

`python3 tests/test_assemble.py` はモデルを載せない。凍結した v0 から本文が残るかだけを見る。

YomiToku は現行の実行依存である。日常の最終形ではない。

## 注意

- 本体のネットは YomiToku 側。このリポジトリは呼び出し順と凍結への導線である。
- YomiToku は CC BY-NC-SA 4.0。商用は本体側のライセンスを見る。
