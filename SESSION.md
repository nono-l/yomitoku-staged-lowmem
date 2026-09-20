# 次のセッションへ

2026-09-20: 検出前後処理を剥いだ。onnx 経路は yomitoku / torch なし。箱は凍結フィクスチャと exact。

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. AR 展開 ONNX を既定にしない。
3. 検出の既定を torch に戻さない。
4. 認識の既定を `stage_recognize.py` に戻さない。
5. 既定の検出・認識が yomitoku を import しないことを戻さない。
6. 実体を git に載せない。

---

## 今動いている面

- 検出既定: `stage_detect.py` + `det/` + 別置き ONNX（yomitoku なし）
- 認識既定: `stage_recognize_onnx.py` + `rec/`（yomitoku なし）
- 剥いだ箱は `tests/fixtures/points_torch_settei21.json` と 37/37 exact
- 既定経路の役: cv2 / numpy / onnxruntime / pyclipper
- shapely は使わない

---

## 次

export スクリプトと torch 比較用だけが yomitoku を残す。
比較用を残すか、export も別プロセスのまま残すか。
