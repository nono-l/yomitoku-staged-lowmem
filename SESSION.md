# 次のセッションへ

2026-09-20: 実体の置き手順を weights/README に現状だけ書いた。runtime と origin を混ぜない。

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

- 既定三段と重みの置き場が README にある
- runtime: detector.onnx + encoder_dynw + decoder_step_dynw + 字表
- origin safetensors は pin_weights だけ

---

## 次

リポジト内の切り出しは止まっている。残るのは cv2/推論の腰と Drive の共有（人がやる）。
