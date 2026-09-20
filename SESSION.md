# 次のセッションへ

2026-09-21: 22 番を分割経路で通した。SOF2 は BMP にしてから。2GiB。

| 段 | 結果 |
|---|---|
| infer / boxes | 20 箱 |
| encode / decode | 20 箱 |
| remain raw | ティーアトリエ / はかりバッジ / ネームタグ「午前」 |
| assemble | タイトル行は score 0.22 で noise |

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

- 21 拡大と 22 は分割経路で 2GiB を生きる
- INTER_AREA は重み行列
- SOF2 は捨てる

---

## 次

assemble が低スコアのタイトル行を noise に隔離する。残す側の契約を raw のままにするか、組み立て側の閾値を見るか。
または重み行列の構築ループを配列化する。
