from __future__ import annotations
import asyncio
import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from yt_dlp import YoutubeDL
from config import get_settings
from file import ensure_dir, sanitize_filename, human_size
import logging  # For better error logs

logger = logging.getLogger(__name__)
_settings = get_settings()

@dataclass
class FormatOption:
    format_id: str
    note: str
    filesize: Optional[int]
    content_type: str
    ext: str
    cached: bool = False
    too_large: bool = False  # > Telegram limit

@dataclass
class MediaInfo:
    provider: str
    title: str
    webpage_url: str
    extractor: str
    duration: Optional[int]
    formats: list[FormatOption]
    thumbnail: Optional[str]

def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()

def _base_ydl_opts(tmp_dir: str, progress_hook: Optional[Callable] = None, format_str: Optional[str] = None) -> dict:
    mobile_headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.snapchat.com/",
    }
    opts: dict[str, Any] = {
        "outtmpl": os.path.join(tmp_dir, "%(id)s_{unique_id}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 10,
        "concurrent_fragment_downloads": 8,
        "merge_output_format": "mp4",
        "http_headers": mobile_headers,
        "allow_unplayable_formats": True,   # 🔑 allow m3u8/dash
        "ignoreerrors": "only_download",    # don’t abort on bad formats
        "postprocessors": [
            {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"},
        ],
    }
    if _settings.ytdlp_cookies_path:
        opts["cookiefile"] = _settings.ytdlp_cookies_path
    if progress_hook:
        opts["progress_hooks"] = [progress_hook]
    if format_str:
        opts["format"] = format_str
    return opts


def _classify_format(fmt: dict) -> tuple[str, str]:
    vcodec = fmt.get("vcodec")
    acodec = fmt.get("acodec")
    ext = fmt.get("ext") or "mp4"
    content_type = "document"
    if vcodec and vcodec != "none" and acodec and acodec != "none":
        content_type = "video"
    elif vcodec and vcodec != "none":
        content_type = "video"
    elif acodec and acodec != "none":
        content_type = "audio"
    return content_type, ext

async def analyze(url: str, cached_formats: set[str]) -> MediaInfo:
    import tempfile
    with tempfile.TemporaryDirectory(dir=_settings.download_dir) as tmp_dir_str:
        tmp_dir = tmp_dir_str
        ydl_opts = _base_ydl_opts(tmp_dir)
        ydl_opts['format'] = None  # Disable format selection for safe metadata extraction (fixes Instagram errors)
        
        # Snapchat-specific retry: Force progressive formats if extractor is SnapchatSpotlight
        is_snapchat = 'snapchat' in url.lower()
        if is_snapchat:
            ydl_opts['format'] = 'best[height<=720]/best'  # Prefer medium, fallback best
        
        def _extract(attempt=1) -> dict:
            with YoutubeDL(ydl_opts) as ydl:
                try:
                    return ydl.extract_info(url, download=False)
                except Exception as e:
                    logger.warning(f"Extraction attempt {attempt} failed: {e}")
                    if attempt == 1 and is_snapchat:
                        # Retry with desktop UA for Snapchat
                        ydl_opts['http_headers']['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                        return _extract(attempt=2)
                    raise e  # Re-raise on final fail
        
        try:
            info = await asyncio.to_thread(_extract)
        except Exception as e:
            logger.error(f"Analyze failed for {url}: {e}")
            # Fallback empty MediaInfo for graceful error
            return MediaInfo(
                provider="unknown",
                title="Failed to analyze",
                webpage_url=url,
                extractor="error",
                duration=None,
                formats=[],
                thumbnail=None
            )
        
        extractor = info.get("extractor_key") or info.get("extractor") or "unknown"
        provider = extractor.lower()
        title = info.get("title") or "Untitled"
        duration = info.get("duration")
        webpage_url = info.get("webpage_url") or url
        thumb = info.get("thumbnail")

        raw_formats = info.get("formats") or []
        max_bytes = _settings.max_telegram_upload_mb * 1024 * 1024

        options: list[FormatOption] = []
        seen = set()
        for f in raw_formats:
            filesize = f.get("filesize") or f.get("filesize_approx")
            content_type, ext = _classify_format(f)

            height = f.get("height")
            abr = f.get("abr")
            fps = f.get("fps")
            vbr = f.get("vbr") or f.get("tbr")
            label_parts = []
            if height:
                label_parts.append(f"{height}p")
            if fps and height:
                label_parts.append(f"{int(fps)}fps")
            if abr and content_type == "audio":
                label_parts.append(f"{int(abr)}kbps")
            if vbr and content_type == "video" and not height:
                label_parts.append(f"{int(vbr)}k")
            label_parts.append(ext)
            if filesize:
                size_label = human_size(filesize)
                if filesize > max_bytes:
                    size_label += " ⚠️>TG"
                label_parts.append(size_label)
            note = " ".join(label_parts)
            fid = str(f.get("format_id"))
            key = (fid, note, content_type, ext)
            if key in seen:
                continue
            seen.add(key)
            options.append(FormatOption(
                format_id=fid,
                note=note,
                filesize=filesize,
                content_type=content_type,
                ext=ext,
                cached=(fid in cached_formats),
                too_large=(filesize and filesize > max_bytes)
            ))

        def sort_key(opt: FormatOption):
            height = 0
            try:
                parts = opt.note.split()
                for p in parts:
                    if p.endswith("p") and p[:-1].isdigit():
                        height = int(p[:-1])
                        break
            except Exception:
                height = 0
            cache_bonus = 1 if opt.cached else 0
            return (-cache_bonus, -height, -(opt.filesize or 0))
        options.sort(key=sort_key)
        options = options[:_settings.max_format_buttons]

        logger.info(f"Extracted {len(options)} formats for {provider}: {title[:50]}...")  # Truncated title for logs

        return MediaInfo(
            provider=provider,
            title=title,
            webpage_url=webpage_url,
            extractor=extractor,
            duration=duration,
            formats=options,
            thumbnail=thumb
        )

async def download(url: str, fmt: FormatOption, tmp_dir: str, progress_cb=None):
    """Download to provided tmp_dir (caller cleans)."""
    unique_id = str(uuid.uuid4()).replace('-', '')  # Full UUID without dashes for filename safety
    ydl_opts = _base_ydl_opts(tmp_dir, progress_hook=progress_cb, format_str=fmt.format_id)
    # Inject unique_id into opts for outtmpl (now shorter, no title)
    ydl_opts['outtmpl'] = ydl_opts['outtmpl'].format(unique_id=unique_id)
    def _run() -> dict:
        with YoutubeDL(ydl_opts) as ydl:
            res = ydl.extract_info(url, download=True)
            return res
    info = await asyncio.to_thread(_run)
    # Find resulting file (now simpler pattern: %(id)s_{unique_id}.%(ext)s)
    ext = fmt.ext or info.get("ext") or "mp4"
    vid_id = info.get('id')
    candidates = list(Path(tmp_dir).glob(f"{vid_id}_{unique_id}.*"))
    p = None
    if candidates:
        # Prefer .mp4 if multiple (e.g., .part then final)
        mp4_cand = [c for c in candidates if c.suffix == '.mp4']
        if mp4_cand:
            p = max(mp4_cand, key=lambda x: x.stat().st_mtime)
        else:
            p = max(candidates, key=lambda x: x.stat().st_mtime)
    if not p or not p.exists():
        requested_filename = info.get("_filename")
        if requested_filename and Path(requested_filename).exists():
            p = Path(requested_filename)
        else:
            raise RuntimeError("Download finished but file not found")
    # Note: Path is returned; caller MUST delete p after sending (e.g., p.unlink() in finally block)
    # The temp_dir is auto-cleaned after this context, but if p is outside (rare), handle manually
    return p, info