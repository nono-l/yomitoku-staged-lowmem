# 次のセッションへ

2026-09-20: README と ocr_raw/v0 に「活字だけ。場面は出さない」を書いた。

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
11. この部品に場面語を混ぜない。

---

## 今動いている面

- 活字の箱・文字列・スコア・読み順
- 場面の説明は別部品

---

## 次

この部品の切り出しは止まっている。次の切は、場面を出す別腰か、使う価の新しい絵か。
