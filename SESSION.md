# 次のセッションへ

2026-09-20: 実行用 ONNX と origin safetensors を 40MiB 未満の parts にして git に載せた。
結合後は weights/pinned/ のまま gitignore。verify は parts から結合してハッシュを見る。

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. AR 展開 ONNX を既定にしない。
3. 検出の既定を torch に戻さない。
4. 認識の既定を torch / yomitoku に戻さない。
5. 既定の検出・認識が yomitoku を import しないことを戻さない。
6. 結合後の実体を git に載せない。50MB 以上の単一ファイルを載せない。
7. 既定 verify に safetensors を強制しない。
8. comparison/ の外で yomitoku / torch を import しない。
9. 既定に pyclipper を戻さない。
10. 裁断なしに imread / resize / 輪郭 / warp / onnxruntime を剥がない。
11. この部品に場面語を混ぜない。
12. README を作業台に戻さない。SESSION を消さない。

---

## 今動いている面

- clone → join_weights.py → verify → run_staged.sh
- 検出 ONNX は凍結ハッシュのまま。認識 dynw は再 export（本文 remain）
- parts は 40MiB 未満

---

## 次

戻すのは使う価の新しい絵か、場面を出す別部品か。
