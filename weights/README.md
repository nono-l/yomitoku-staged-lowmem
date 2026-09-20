# 重みの parts

GitHub は 100MB を拒み、50MB で警告する。実行用の実体は 40MiB 未満に切って `weights/parts/` に置く。
parts の合計は約 270MB。結合後は `weights/pinned/`（gitignore）。この目録に yomitoku は置かない。

```
python3 weights/join_weights.py
python3 weights/verify_weights.py
```

無い・違うと失敗する。ネットへ救済しに行かない。

## 実行に要るもの

`manifest.json` で `runtime: true` のもの。join が書く。

| 役 | 結合後 |
|---|---|
| 検出 ONNX | `weights/pinned/detector/model.onnx` |
| 認識 encoder（幅動的） | `weights/pinned/recognizer/encoder_dynw.onnx` |
| 認識 decoder 1 ステップ | `weights/pinned/recognizer/decoder_step_dynw.onnx` |
| 字表・置換 | `rec/resource/` （git にある。切らない） |

parts を切り直すときは `python3 weights/split_weights.py`。結合後のハッシュが manifest と一致しているときだけ切る。

## 比較・再 export 用

origin の safetensors も parts にある。既定の入口では使わない。

```
python3 weights/join_weights.py --origin
python3 weights/verify_weights.py --origin
```

Drive は予備: https://drive.google.com/drive/folders/1HN5R00gE_wa0SuzZAaTqA6RJyaf_4J-G
export は `comparison/export/`。
