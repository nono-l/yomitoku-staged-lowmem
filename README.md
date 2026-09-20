# yomitoku-staged-lowmem

画素から図中の活字を、箱・文字列・スコア・読み順で出す部品。分野は知らない。

人間と、あとから入った AI の両方が読める説明。
方針は [CODING.md](CODING.md)。次のセッションへの記録は [SESSION.md](SESSION.md)。

## 今動く面

[YomiToku](https://github.com/kotaro-kinoshita/yomitoku) の検出と認識を、別プロセスで回す。
作業台に両方を同時に広げない。約 2GiB RAM・swap なしでフル CLI が 137 になるため。

```bash
pip install yomitoku==0.15.0
chmod +x run_staged.sh stage_detect.py stage_recognize.py
./run_staged.sh path/to/image.jpg results
```

YomiToku は現行の実行依存である。日常の最終形ではない。進め方は SESSION.md の順。

## 注意

- 本体のネットは YomiToku 側。このリポジトリは呼び出し順と凍結への導線である。
- YomiToku は CC BY-NC-SA 4.0。商用は本体側のライセンスを見る。
