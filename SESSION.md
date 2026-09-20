# 次のセッションへ

2026-09-20: yomitoku / torch の import を comparison/ に閉じ込めた。export もそこへ移した。

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

---

## 今動いている面

- 既定: cv2 / numpy / onnxruntime / pyclipper
- yomitoku: `comparison/` だけ
- export: `comparison/export/`

---

## 次

残るのは pyclipper（箱の offset）。剥ぐなら箱が exact のまま残すこと。
