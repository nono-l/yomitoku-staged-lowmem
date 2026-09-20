"""字表と greedy 復号。torch は使わない。"""

from __future__ import annotations

import csv
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CHARSET_PATH = os.path.join(ROOT, "resource", "charsetv3.txt")
REPLACE_PATH = os.path.join(ROOT, "resource", "character_post_expand_table_v3.csv")


def load_charset(path=CHARSET_PATH):
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_replace_table(path=REPLACE_PATH):
    table = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source = row["source"]
            table[ord(source)] = row["target"]
    return table


class GreedyTokenizer:
    EOS = "[E]"
    BOS = "[B]"
    PAD = "[P]"

    def __init__(self, charset: str):
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}
        self.eos_id = self._stoi[self.EOS]
        self.bos_id = self._stoi[self.BOS]
        self.pad_id = self._stoi[self.PAD]

    def decode_one(self, probs_tc) -> tuple[str, float]:
        ids = probs_tc.argmax(-1).tolist()
        pmax = probs_tc.max(axis=-1)
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)
        ids = ids[:eos_idx]
        kept = pmax[: eos_idx + 1]
        text = "".join(self._itos[i] for i in ids)
        score = float(kept.prod()) if len(kept) else 0.0
        return text, score
