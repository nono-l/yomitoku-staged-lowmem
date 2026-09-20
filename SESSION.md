# 次のセッションへ

2026-09-20: README を現状の既定経路に合わせた。分野語は入れていない。

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

- README が既定の三段（detect / rec / assemble）を書く
- 既定: cv2 / numpy / onnxruntime
- 箱 37/37 exact

---

## 次

部品の腰は残した。次に手を出すなら、使う人が実体を置く手順を weights/README に現状だけ書く余地。
