#!/bin/sh
# 検出の前に safetensors の別置きを見る。
# 検出の既定は別置き ONNX（stage_detect.py）。認識はまだ旧面。
# 検出プロセスが死んでから認識を起こす。
set -eu

IMAGE=${1:?usage: ./run_staged.sh IMAGE [outdir]}
OUTDIR=${2:-results}

python3 weights/verify_weights.py

mkdir -p "$OUTDIR"
python3 stage_detect.py "$IMAGE" -o "$OUTDIR/points.json"
python3 stage_recognize.py "$IMAGE" --points "$OUTDIR/points.json" -o "$OUTDIR/ocr.json"
python3 stage_assemble.py "$OUTDIR/ocr.json" -o "$OUTDIR/document.json"
