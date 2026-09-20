# 次のセッションへ

2026-09-20: torch / yomitoku 実行を comparison/ に後退した。既定は yomitoku を import しない。

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. AR 展開 ONNX を既定にしない。
3. 検出の既定を torch に戻さない。
4. 認識の既定を torch / yomitoku に戻さない。
5. 既定の検出・認識が yomitoku を import しないことを戻さない。
6. 実体を git に載せない。

---

## 今動いている面

- 既定: `run_staged.sh` → detect ONNX → rec ONNX → assemble
- 役: cv2 / numpy / onnxruntime / pyclipper
- 比較用: `comparison/` と `weights/export_*.py`
- テスト: `tests/test_default_no_yomitoku.py` で既定の import を見る

---

## 次

export は別プロセスのまま残す。
verify が safetensors を強制するなら、既定は graphs 側に寄る。
