#!/bin/sh
# 検出の前に、実行に使う ONNX のハッシュを見る。safetensors は見ない。
# 検出の既定は別置き ONNX。認識の既定は stage_recognize_onnx.py。
# torch 比較は comparison/ 。このスクリプトの既定に戻さない。
# 検出プロセスが死んでから認識を起こす。
set -eu

IMAGE=${1:?usage: ./run_staged.sh IMAGE [outdir]}
OUTDIR=${2:-results}

python3 weights/verify_weights.py

mkdir -p "$OUTDIR"
python3 stage_detect.py "$IMAGE" -o "$OUTDIR/points.json"
python3 stage_recognize_onnx.py "$IMAGE" --points "$OUTDIR/points.json" -o "$OUTDIR/ocr.json"
python3 stage_assemble.py "$OUTDIR/ocr.json" -o "$OUTDIR/document.json"
