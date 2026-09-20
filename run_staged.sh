#!/bin/sh
# 検出の前に別置きを見る。無いなら止める。黙って latest を引かない。
# 検出プロセスが死んでから認識を起こす。
# 組み立ては重みを載せない。
set -eu

IMAGE=${1:?usage: ./run_staged.sh IMAGE [outdir]}
OUTDIR=${2:-results}

python3 weights/verify_weights.py

mkdir -p "$OUTDIR"
python3 stage_detect.py "$IMAGE" -o "$OUTDIR/points.json"
python3 stage_recognize.py "$IMAGE" --points "$OUTDIR/points.json" -o "$OUTDIR/ocr.json"
python3 stage_assemble.py "$OUTDIR/ocr.json" -o "$OUTDIR/document.json"
