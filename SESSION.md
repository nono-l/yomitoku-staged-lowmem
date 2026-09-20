# 次のセッションへ

2026-09-20: 箱数の差は輪郭ではなく INTER_AREA。
公式画素を BMP で渡しても、縮小の平均の取り方が違う。

| 経路 | 箱 |
|---|---|
| 旧公式（fixture） | 37 |
| 公式縮小 + 既定後処理 | 36 |
| 既定縮小 + 既定後処理 | 29 |
| 同一 bitmap の輪郭数 | 公式 67 / 既定 69 |

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

- remain 3 文は 29〜36 箱でも残る
- SOF2 は捨てる
- INTER_AREA の端重みが未実装

---

## 次

INTER_AREA に端の分数重みを入れて、公式縮小に近づける。
提出前に settei21 で箱数を見る。
