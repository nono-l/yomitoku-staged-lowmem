# 次のセッションへ

2026-09-20: pin_weights は origin だけ見る。offset の箱をモデルなしで凍結した。
cv2 / onnxruntime は剥がない。JPEG・輪郭・warp を置き換えると exact が死ぬ。

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

- 既定: cv2 / numpy / onnxruntime
- pin_weights: origin safetensors
- runtime verify: graphs + charset
- offset ケース 3 件がモデルなしで緑

---

## 次

cv2 と onnxruntime は腰。剥ぐなら exact が残る裁断を先に書く。
