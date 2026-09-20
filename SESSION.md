# 次のセッションへ

2026-09-20: 基線 JPEG の小紙を試験に埋めた。SOF2 は明示に捨てる。
settei21 の基線化で検出 31 箱（BMP 30 / 旧 37）。

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

- 基線 JPEG 16x8 を読める
- SOF2 は捨てる
- settei21 remain 3 文は残る

---

## 次

箱数 31 と 37 の差を、輪郭と補間のどちらが動かしているかに分ける。
