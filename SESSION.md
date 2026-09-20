# 次のセッションへ

2026-09-20: settei21 を BMP にして remain を取り直した。箱は 30（旧 37）。must_remain 3 文は本文に残る。
原 JPEG は SOF2。基線以外は明示に捨てる。pos_queries は onnx パッケージなし。

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

---

## 今動いている面

- 既定 import は `from cvsurf import cv2`
- settei21 remain: 夜光餃子座 / パートタイマー制服資料 / 光る餃子座ピン
- JPEG は基線 SOF0 のみ

---

## 次

プログレッシブ JPEG を支えるか、試験用の基線フィクスチャを置く。
