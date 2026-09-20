# 別置き重み

git にはハッシュだけを置く。実体は `pinned/` か Drive である。

```
python3 weights/pin_weights.py
python3 weights/verify_weights.py
python3 weights/export_detector_onnx.py
```

検出の既定は `weights/pinned/detector/model.onnx`。
Drive: https://drive.google.com/drive/folders/1HN5R00gE_wa0SuzZAaTqA6RJyaf_4J-G
