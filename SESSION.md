# 次のセッションへ

2026-09-21: 輪郭の開始点を配列で拾う。全画素の二重ループは止めた。
1184×1600 に帯 2 本で 0.01 秒。

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

- 輪郭開始は左縁だけ
- 輪郭追跡に上限
- 検出は infer → boxes
- INTER_AREA / warp 最終行 / SOF2 捨て

---

## 次

認識は encoder と decoder を同じプロセスに載せている。
離す必要が残るか、箱ごとに encoder を捨てるだけで足りるか見る。
