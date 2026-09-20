# 次のセッションへ

2026-09-20: cv2.boxPoints を剥いだ。箱は exact。
残る cv2 は imread / resize / 輪郭 / fillPoly / warp。

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

---

## 今動いている面

- 既定: cv2（imread/resize/輪郭/fillPoly/warp）/ numpy / onnxruntime
- boxPoints / rotate / mean は numpy
- 箱 37/37 exact

---

## 次

残る cv2 と onnxruntime は腰。JPEG・輪郭・warp を剥ぐ裁断はまだない。
