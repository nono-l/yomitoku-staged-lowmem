# 次のセッションへ

2026-09-21: 認識の encoder と decoder を別プロセスにした。
stage_recognize_onnx.py は入口だけ。グラフは同時に載せない。

| 段 | 入口 | 出口 |
|---|---|---|
| encode | 画像 + points | memory.npz |
| decode | memory.npz + points | ocr.json |
| gate | 画像 + points | 上を順に起こす |

import 分割の試験は緑。重みを載せる往復はこの箱ではしていない。

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
10. 既定経路に opencv-python を戻さない。画素は cvsurf。
11. この部品に場面語を混ぜない。
12. README を作業台に戻さない。SESSION を消さない。
13. 検出の推論と箱出しを同じプロセスに戻さない。
14. 認識の encoder と decoder を同じプロセスに戻さない。

---

## 今動いている面

- 検出 infer → boxes
- 認識 encode → decode
- 輪郭開始は左縁、追跡に上限
- INTER_AREA / warp 最終行 / SOF2 捨て

---

## 次

結合した重みがある箱で、encode のあとに decoder が起きるか一度通す。
または INTER_AREA の純 Python 縮小が遅いときの切り出し。
