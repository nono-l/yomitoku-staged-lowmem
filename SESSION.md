# 次のセッションへ

2026-09-20: charset / replace table を runtime pin に載せた。

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. AR 展開 ONNX を既定にしない。
3. 検出の既定を torch に戻さない。
4. 認識の既定を torch / yomitoku に戻さない。
5. 既定の検出・認識が yomitoku を import しないことを戻さない。
6. 実体を git に載せない。
7. 既定 verify に safetensors を強制しない。

---

## 今動いている面

- runtime pin: detector.onnx + encoder_dynw + decoder_step_dynw + charsetv3 + replace table
- charset を 1 バイト足すと verify が落ちる
- assemble / import 守りは緑

---

## 次

export は別プロセスのまま。
残るのは comparison/ と export だけ yomitoku に依存すること。
