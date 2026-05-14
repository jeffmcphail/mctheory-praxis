"""Decode UTF-16-LE phase4 logs and write plain-UTF-8 sibling copies."""
from pathlib import Path
import re

for cap in ["2", "1", "0.5", "0.25"]:
    src = Path(f"outputs/exp10_revival/cap_{cap}/phase4.log")
    if not src.exists():
        print(f"missing: {src}")
        continue
    raw = src.read_bytes()
    if raw[:2] == b"\xff\xfe":
        txt = raw.decode("utf-16-le", errors="replace")
    else:
        txt = raw.decode("utf-16-le", errors="replace").replace("\x00", "")
    txt = txt.replace("\x00", "")
    out = src.with_suffix(".txt")
    out.write_text(txt, encoding="utf-8")
    print(f"wrote: {out}")
