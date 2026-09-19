#!/bin/sh
# Two-process pipeline: detector dies before recognizer starts.
set -eu

IMAGE=${1:?usage: ./run_staged.sh IMAGE [outdir]}
OUTDIR=${2:-results}

mkdir -p "$OUTDIR"
python3 stage_detect.py "$IMAGE" -o "$OUTDIR/points.json"
python3 stage_recognize.py "$IMAGE" --points "$OUTDIR/points.json" -o "$OUTDIR/ocr.json"
