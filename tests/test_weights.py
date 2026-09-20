#!/usr/bin/env python3
"""別置きが無いときは落ちる。ハッシュが違うときも落ちる。ネットへ行かない。
既定 verify は runtime graphs と字表。safetensors は見ない。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "weights"))
from verify_weights import runtime_graphs, runtime_items, verify


def main() -> None:
    with open(os.path.join(ROOT, "weights", "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    items = runtime_items(man)
    if not any(i["id"] == "charset_v3" for i in items):
        raise SystemExit("charset_v3 missing from runtime pin")
    if not any(i["id"] == "char_replace_v3" for i in items):
        raise SystemExit("char_replace_v3 missing from runtime pin")
    print("ok charset in runtime pin")

    with tempfile.TemporaryDirectory() as tmp:
        man_src = os.path.join(ROOT, "weights", "manifest.json")
        dest_man = os.path.join(tmp, "weights")
        os.makedirs(dest_man)
        shutil.copy(man_src, os.path.join(dest_man, "manifest.json"))
        errors = verify(tmp, what="runtime")
        if not errors:
            raise SystemExit("missing runtime pin should fail")
        if not any(e.startswith("missing ") for e in errors):
            raise SystemExit("expected missing, got: " + "; ".join(errors))
        print("ok missing runtime fails")

        graphs = runtime_graphs(man)
        if not graphs:
            raise SystemExit("manifest has no runtime graphs")
        first = graphs[0]
        fake = os.path.join(tmp, first["path"])
        os.makedirs(os.path.dirname(fake), exist_ok=True)
        with open(fake, "wb") as f:
            f.write(b"not-the-graph")
        errors = verify(tmp, what="runtime")
        if not errors:
            raise SystemExit("bad hash should fail")
        print("ok mismatch fails")

        errors_origin = verify(tmp, what="origin")
        if not errors_origin:
            raise SystemExit("origin should still want safetensors")
        print("ok origin still checks files")

    live = os.path.join(ROOT, "weights", "pinned", "detector", "model.onnx")
    if os.path.isfile(live):
        errors = verify(ROOT, what="runtime")
        if errors:
            raise SystemExit("live runtime pin broken: " + "; ".join(errors))
        print("ok live runtime pin")
    else:
        # 字表だけでも欠けるなら字表の pin が壊れている
        errors = verify(ROOT, what="runtime")
        charset_err = [e for e in errors if "charset" in e or "character_post" in e]
        if charset_err:
            raise SystemExit("charset pin broken: " + "; ".join(charset_err))
        print("skip live graph pin (not in this tree); charset ok")


if __name__ == "__main__":
    main()
