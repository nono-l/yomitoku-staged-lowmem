# 次のセッションへ

2026-09-21: 検出の推論と箱出しを別プロセスにした。
stage_detect.py は入口だけで、onnx も後処理も import しない。

| 段 | 入口 | 出口 |
|---|---|---|
| infer | 画像 | pred.npz（float16） |
| boxes | pred.npz | points.json |
| gate | 画像 | 上を順に起こす |

偽予測図で箱 1 つ出る試験は緑。

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

---

## 今動いている面

- 輪郭追跡に上限
- INTER_AREA は重なり面積
- warp は最終行を残す
- SOF2 は捨てる
- 検出は infer → boxes

---

## 次

輪郭の画素二重ループを削る。または認識も encoder と decoder を今以上に離す必要が残るか見る。
