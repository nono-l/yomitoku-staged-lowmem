# 次のセッションへ

2026-09-20: pyclipper を剥いだ。箱は凍結フィクスチャと exact。

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
- offset: `det/offset.py`（Clipper 6.4.2 JT_ROUND 相当）
- 箱 37/37 exact vs points_torch_settei21.json

---

## 次

残る外部役は cv2 と onnxruntime。
そこを剥ぐなら、箱も活字も exact のまま。
