# 次のセッションへ

コメント方針は [CODING.md](CODING.md)。

2026-09-20 App Builder: 重み実体を Drive へ別置き。git には載せていない。
次は C3（検出の既定を ONNX）。

---

## これは何か

画素から図中の活字を、箱と信頼度と読み順つきで取り出す部品。
https://github.com/nono-l/yomitoku-staged-lowmem

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. 分野語をスキーマに入れない。
3. DBNet / PARSeq を今夜書き直さない。
4. latest を黙って引かない。
5. 提出してから動かさない。
6. safetensors / onnx 実体を git に載せない。
7. T2 が緑でも C3 の前に検出既定を黙って ONNX にしない。

---

## 別置き（Drive）

フォルダ: https://drive.google.com/drive/folders/1HN5R00gE_wa0SuzZAaTqA6RJyaf_4J-G

- detector 98MB `1b3mvWpwfnQvBn6EE0k1z6n8cj2ZWZhCL` sha256 fa090966…
- recognizer 35MB `1ukXNL3Db5iHAfcMXtL0Sh9P3IlDBTpeA` sha256 498db515…
- v5_manifest `1SCcQKVnHQMwqmesVsNH0olBsHi9C3iHL`

正本の入手はまだ HF revision（pin_weights.py）。Drive は容量用のコピー。

---

## 重みが RAM に載らないとき

今の 98MB+35MB は、torch を載せた後の「二つ同時」に比べて小さい。
分割ロードは可能だが、YomiToku のスイッチではない。次の階:
1. プロセス分割（済）
2. ONNX mmap
3. バックボーンとヘッドを別グラフにし、中間激活を盤へ落とす
層ごとに重みを切って順に matmul するのは 3 の後。激活地図が重みより重い。

---

## 順

| ID | 状態 |
|---|---|
| C0–T2 / P1b | 済 |
| C3 | 次。検出既定 ONNX |
| P1c | 認識 ONNX |
