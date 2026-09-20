# 次のセッションへ

2026-09-20: 認識から TextRecognizer / 36MB torch 重みを外した比較用経路。C4 しない。

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. AR 展開 ONNX を既定にしない。
3. 検出の既定を torch に戻さない。
4. `stage_recognize_onnx.py` を `run_staged.sh` の既定にしない。
5. 実体を git に載せない。

---

## 今動いている面

- 検出既定: 別置き ONNX
- 認識既定: torch tiny dynamic_width
- 比較: `stage_recognize_onnx.py`（重みなし、検出順、幅は箱ごと）
- vs 既定 rec: 4 行違い（`11`/`1.` `44`/`48` `18.18`/`18.11` `。」`/`。`）。本文 3 つ残る
- 差はバッチ垫きをしないため。本文ではない

---

## 次

C4 はまだしない。切るなら、端の 4 行を許して `run_staged.sh` の認識を `stage_recognize_onnx.py` にする。
先に Dataset / Tokenizer を yomitoku から剥ぐかは別件。
