# 次のセッションへ

2026-09-20: C4 済。認識既定を `stage_recognize_onnx.py` に切った。

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. AR 展開 ONNX を既定にしない。
3. 検出の既定を torch に戻さない。
4. 認識の既定を `stage_recognize.py`（torch）に戻さない。
5. 実体を git に載せない。

---

## 今動いている面

- 検出既定: 別置き ONNX
- 認識既定: `stage_recognize_onnx.py`（encoder_dynw + decoder_step_dynw、TextRecognizer なし）
- torch 認識: `stage_recognize.py` 比較用
- C4 提出前: lite 活字を assemble し、夜光餃子座 / パートタイマー制服資料 / 光る餃子座ピン が body に残る
- 既定 rec と 4 行違う。端の数字・句読。許した

---

## 次

Dataset / Tokenizer を yomitoku から剥ぐ。
とりあえず既定経路が ONNX だけで回ることを優先する。
