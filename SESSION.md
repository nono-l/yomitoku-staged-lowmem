# 次のセッションへ

コメント方針は [CODING.md](CODING.md)。

2026-09-20: C3 済。検出の既定は別置き ONNX。認識はまだ tiny torch。
次は P1c（認識 ONNX 書き出し。既定はまだ旧認識）。

---

## これは何か

画素から図中の活字を取る部品。
https://github.com/nono-l/yomitoku-staged-lowmem

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. 分野語をスキーマに入れない。
3. DBNet / PARSeq を今夜書き直さない。
4. latest を黙って引かない。
5. 提出してから動かさない。
6. safetensors / onnx 実体を git に載せない。
7. 検出の既定を torch に戻さない。比較は `--backend torch`。
8. ONNX 検出で TextDetector / from_pretrained を使わない。

---

## 今動いている面

- `stage_detect.py` 既定 backend=onnx
- 実測 settei/21: 37 箱、旧 torch フィクスチャと exact
- ONNX sha256 `1236c4c1761956d91f7ac7c897983d8b062320c9c62d7d36b0c54e29b05e21f4` 101894163 bytes
- Drive: https://drive.google.com/drive/folders/1HN5R00gE_wa0SuzZAaTqA6RJyaf_4J-G

---

## 順

| ID | 状態 |
|---|---|
| C0–C3 / T2 | 済 |
| P1c | 次。認識 ONNX |
| T3 | 旧認識と ONNX 認識 |
| C4 | 認識既定 ONNX。torch 外し |
