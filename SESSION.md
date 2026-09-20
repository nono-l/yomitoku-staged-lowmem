# 次のセッションへ

2026-09-20: P1d / T4 まで。認識既定はまだ torch tiny。C4 しない。

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. PARSeq AR を 100 歩展開した ONNX を既定にしない。
3. 検出の既定を torch に戻さない。
4. 二グラフ認識を順序未復元のまま既定にしない。
5. 実体を git に載せない。

---

## 今動いている面

- 検出既定: 別置き ONNX
- 認識既定: torch tiny dynamic_width
- encoder ONNX 22MB `e6eab9c3…`
- decoder_step ONNX 14MB `79121be5…` バッチ=1、len は 2–101
- T4: 本文 3 つ残る。torch800 と袋として 1000/1200 の 1 件だけ違う。行順序は未復元。

Drive: https://drive.google.com/drive/folders/1HN5R00gE_wa0SuzZAaTqA6RJyaf_4J-G

---

## 次

C4 ではない。先に二グラフループの順序を検出順に戻す。
それが緑で、端の数字ずれを許すなら、比較用バックエンドにできる。既定はまだ別。
