"""decoder 1 ステップ。encoder は載せない。"""

from __future__ import annotations

import numpy as np


def load_pos_queries(path):
    """decoder ONNX の initializer から取る。onnx パッケージは使わない。"""
    data = open(path, "rb").read()
    key = b"B\x0bpos_queriesJ"
    j = data.find(key)
    if j < 0:
        raise SystemExit(f"pos_queries missing in {path}")
    i = j + len(key)
    n = 0
    shift = 0
    while True:
        by = data[i]
        i += 1
        n |= (by & 0x7F) << shift
        if by < 0x80:
            break
        shift += 7
    arr = np.frombuffer(data[i : i + n], dtype=np.float32)
    if arr.size != 1 * 101 * 192:
        raise SystemExit(f"pos_queries size {arr.size}")
    return arr.reshape(1, 101, 192)


def logits_from_memory(dec, pos_queries, bos, pad, eos, num_steps, memory):
    tgt = np.full((1, num_steps), pad, dtype=np.int64)
    tgt[:, 0] = bos
    mask = np.triu(np.ones((num_steps, num_steps), dtype=bool), 1)
    steps = []
    for i in range(num_steps):
        j = i + 1
        logits = dec.run(
            ["logits"],
            {
                "memory": memory,
                "tgt": tgt[:, :j],
                "pos_query": pos_queries[:, i:j],
                "query_mask": mask[i:j, :j],
            },
        )[0]
        steps.append(logits)
        if j < num_steps:
            nxt = int(logits[0, 0].argmax())
            tgt[0, j] = nxt
            if nxt == eos:
                break
    p = np.concatenate(steps, axis=1)
    p = np.exp(p - p.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    return p[0]
