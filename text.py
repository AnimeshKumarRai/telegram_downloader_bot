from __future__ import annotations
import re

URL_REGEX = re.compile(r"(https?://[^\s]+)")

def extract_urls(text: str) -> list[str]:
    return URL_REGEX.findall(text or "")