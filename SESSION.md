# 次のセッションへ

コメント方針は [CODING.md](CODING.md)。

このセッション（2026-09-20）: P1b / T2 まで緑。
次は C3（検出の既定を ONNX にする）。認識の既定はまだ旧面。

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
7. T2 が緑でも、C3 の前に検出の既定を黙って ONNX にしない。載せ替えは次の切。

---

## 今動いている面

- 既定入口: `run_staged.sh` はまだ torch 検出 + tiny 認識 + assemble
- 検出 ONNX 書き出し: `python3 weights/export_detector_onnx.py`
- T2: `python3 tests/test_detect_compare.py`

実測 2026-09-20 settei/21: 旧検出と ONNX 検出は 37 箱、exact 一致、min IoU 1.0。
opset は 16 指定が 18 に落ちる。単一ファイル ONNX 101894186 bytes、sha256 `01b00d6523112c577a33d025326596d96720cd5d89e75a06a74e1775ec4c8add`。

---

## 順

| ID | 状態 |
|---|---|
| C0–C2 / P0–P1a | 済 |
| P1b | 済 書き出しスクリプト |
| T2 | 済 exact |
| C3 | 次。検出既定を ONNX |
| P1c | 認識 ONNX |

---

## 既知の粗

- convert_onnx は外部 .data を出す。export スクリプトが単一ファイルへ踏む。
- YomiToku infer_onnx はパッケージ内 onnx/ を見る。C3 で別置きパスを既定にする。
- サンドボックス 1.9GiB。
