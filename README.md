# yomitoku-staged-lowmem

[YomiToku](https://github.com/kotaro-kinoshita/yomitoku) を、検出と認識でプロセスを分けて回す薄いラッパです。

作業台に検出器と認識器を同時に広げない。約 2GiB RAM・swap なしの環境で、フル CLI が 137 で落ちるところまで確認したうえで、二段に刻んだら通った、という実測が起点です。

## なぜ分けるか

メモリは作業台です。全部を常に広げる必要はありません。

1. `stage_detect.py` — `TextDetector` だけを載せる。箱座標を JSON に書いて終了する。
2. `stage_recognize.py` — 新しいプロセスで `parseq-tiny-dynw-v5` だけを載せる。箱を 6 個ずつ認識する。

フルの `yomitoku` CLI（レイアウト解析と表構造まで同時）はこの分割の対象外です。店名など図中活字の有無を見る用途向けです。

## 使い方

```bash
pip install yomitoku==0.15.0
chmod +x run_staged.sh stage_detect.py stage_recognize.py
./run_staged.sh path/to/image.jpg results
```

または:

```bash
python3 stage_detect.py path/to/image.jpg -o results/points.json
python3 stage_recognize.py path/to/image.jpg --points results/points.json -o results/ocr.json
```

## 実測

`https://cc0-kisaragi.grok.me/settei/21/card.jpg`（800×600）で検出 37 箱。認識結果に次が残った。

- `クター|中華料理店「夜光餃子座」パートタイマー制服資料`
- `「夜光」の文字入り。`
- `· 光る餃子座ピン`

低スコアの数字断片はごみです。固有名側は落ちていません。

## 注意

- 本体は YomiToku です。このリポジトリは呼び出し順だけを固定します。
- YomiToku は CC BY-NC-SA 4.0 です。商用は本体側のライセンスを見てください。
- モデル初回実行時に Hugging Face から重みが落ちます。
- CPU 専用。スレッド数は 1 に固定しています。
