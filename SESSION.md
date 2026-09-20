# 次のセッションへ

2026-09-20: 残る cv2 の exact 裁断を `tests/test_cv2_surface.py` に書いた。剥がない。

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. AR 展開 ONNX を既定にしない。
3. 検出の既定を torch に戻さない。
4. 認識の既定を torch / yomitoku に戻さない。
5. 既定の検出・認識が yomitoku を import しないことを戻さない。
6. 実体を git に載せない。
7. 既定 verify に safetensors を強制しない。
8. comparison/ の外で yomitoku / torch を import しない。
9. 既定に pyclipper を戻さない。
10. 裁断なしに imread / resize / 輪郭 / warp / onnxruntime を剥がない。

---

## 今動いている面

- 既定: cv2（裁断済みの残り）/ numpy / onnxruntime
- 剥いだ: rotate / mean / boxPoints / offset / charset pin
- 箱 37/37 exact

---

## 次

腰は残した。次は分野側（店名の組み立て）ではなく、この部品の使い方を README に現状通り書く余地。
