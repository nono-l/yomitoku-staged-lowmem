# 次のセッションへ

2026-09-20: 既定 verify を runtime ONNX に寄せた。safetensors は `--origin`。

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

- `python3 weights/verify_weights.py` → detector.onnx + encoder_dynw + decoder_step_dynw
- `--origin` だけ safetensors
- 活字の assemble と import 守りは緑

---

## 次

export は別プロセスのまま。
charset もハッシュを manifest に載せる余地がある。
