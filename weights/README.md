# 別置き重み

git にはハッシュだけを置く。実体は `weights/pinned/` か Drive である。
この目録に yomitoku は置かない。

Drive: https://drive.google.com/drive/folders/1HN5R00gE_wa0SuzZAaTqA6RJyaf_4J-G

## 実行に要るもの

`manifest.json` で `runtime: true` のもの。

| 役 | 置く場所 |
|---|---|
| 検出 ONNX | `weights/pinned/detector/model.onnx` |
| 認識 encoder（幅動的） | `weights/pinned/recognizer/encoder_dynw.onnx` |
| 認識 decoder 1 ステップ | `weights/pinned/recognizer/decoder_step_dynw.onnx` |
| 字表・置換 | `rec/resource/` （git にある） |

手順:

1. Drive から上の三つの ONNX を下ろす。
2. 上の場所へ名前そのまま置く。
3. `python3 weights/verify_weights.py`

無い・差違うと失敗する。ネットへ救済しに行かない。

## 比較・再 export 用

origin の safetensors は既定の入口に使わない。

```
python3 weights/pin_weights.py
python3 weights/verify_weights.py --origin
```

`pin_weights.py` は manifest の `files` を HF の revision から取る。graphs は取らない。
export は `comparison/export/`。
