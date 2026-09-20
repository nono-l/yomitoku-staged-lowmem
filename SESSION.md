# 次のセッションへ

2026-09-20: INTER_AREA に端の分数重み。拡大でも線形に落とさない。
settei21 の縮小差は 3e-5。公式縮小も既定縮小も 36 箱。

| 経路 | 箱 |
|---|---|
| 旧公式 fixture | 37 |
| 公式 INTER_AREA + 既定後処理 | 36 |
| 既定 INTER_AREA + 既定後処理 | 36 |

残り 1 箱は後処理（輪郭 / minAreaRect）。

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

- INTER_AREA は重なり面積。整数倍拡大は画素の繰り返し
- remain 3 文は残る
- SOF2 は捨てる

---

## 次

36 と 37 の 1 箱を、輪郭採取と minAreaRect のどちらが落とすか見る。
