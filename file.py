from __future__ import annotations
import os
import re
from pathlib import Path
from slugify import slugify

def sanitize_filename(name: str, ext: str | None = None) -> str:
    base = slugify(name, lowercase=False, max_length=150)
    if not base:
        base = "file"
    if ext:
        ext = re.sub(r"[^a-zA-Z0-9]", "", ext)
        return f"{base}.{ext}" if ext else base
    return base

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def human_size(num: int | float, suffix="B") -> str:
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"