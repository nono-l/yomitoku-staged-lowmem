#!/usr/bin/env python3
"""別置きが無いときは落ちる。ハッシュが違うときも落ちる。ネットへ行かない。"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "weights"))
from verify_weights import verify


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        man_src = os.path.join(ROOT, "weights", "manifest.json")
        dest_man = os.path.join(tmp, "weights")
        os.makedirs(dest_man)
        shutil.copy(man_src, os.path.join(dest_man, "manifest.json"))
        errors = verify(tmp)
        if not errors:
            raise SystemExit("missing pin should fail")
        if not any(e.startswith("missing ") for e in errors):
            raise SystemExit("expected missing, got: " + "; ".join(errors))
        print("ok missing fails")

        with open(man_src, encoding="utf-8") as f:
            man = json.load(f)
        first = man["files"][0]
        fake = os.path.join(tmp, first["path"])
        os.makedirs(os.path.dirname(fake), exist_ok=True)
        with open(fake, "wb") as f:
            f.write(b"not-the-weights")
        errors = verify(tmp)
        if not errors:
            raise SystemExit("bad hash should fail")
        print("ok mismatch fails")

    live = os.path.join(ROOT, "weights", "pinned", "detector", "model.safetensors")
    if os.path.isfile(live):
        errors = verify(ROOT)
        if errors:
            raise SystemExit("live pin broken: " + "; ".join(errors))
        print("ok live pin")
    else:
        print("skip live pin (not in this tree)")


if __name__ == "__main__":
    main()
