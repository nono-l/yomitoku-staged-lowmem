# 次のセッションへ

コメント方針は [CODING.md](CODING.md)。

2026-09-20: P1c / T3 済。認識の encoder だけ ONNX。既定認識はまだ torch tiny。
次は C4 ではない。先に認識既定を混成経路にするか、デコーダ 1 ステップを別グラフにするかを決める。
全量 AR 展開の convert_onnx は使わない。

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. 分野語をスキーマに入れない。
3. PARSeq の AR を 100 歩展開した ONNX を既定にしない。
4. latest を黙って引かない。
5. 検出の既定を torch に戻さない。
6. 実体を git に載せない。

---

## 今動いている面

- 検出既定: 別置き ONNX
- 認識既定: torch tiny（dynamic_width）
- encoder ONNX 22MB、sha256 `e6eab9c3266a6f986c5776929ec783e4fddec8e5e0325bc9fcbf503cc46d5b73`
- T3: 32x800 torch と encoder-ONNX 混成が 37 行 exact、夜光餃子座が残る
- Drive: https://drive.google.com/drive/folders/1HN5R00gE_wa0SuzZAaTqA6RJyaf_4J-G

重み内訳（tiny）: encoder 22MB / decoder 2.4MB / head 5.8MB / text_embed 5.8MB。
分割するなら encoder が先。デコーダの展開ではない。

---

## 順

| ID | 状態 |
|---|---|
| C0–C3 / P1c / T3 | 済 |
| 次 | 認識混成を比較用スクリプトに残すか、C4 で既定にするか |
