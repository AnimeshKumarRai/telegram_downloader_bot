from __future__ import annotations

import asyncio
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    InputFile,
)   
from telegram.ext import ContextTypes
from sqlalchemy import select

from config import get_settings
from logger import setup_logging
from downloader import analyze, download, MediaInfo, FormatOption, url_hash
from text import extract_urls
from file import human_size
from rate_limit import rate_limiter
from database import SessionLocal
from models import CacheItem, JobLog
from redis_client import get_json as redis_get_json, set_json as redis_set_json, delete_json
import tempfile
from contextlib import contextmanager
import shutil

logger = setup_logging()
_settings = get_settings()

@contextmanager
def temp_download_dir(base_dir=None):
    """Context manager for temporary download directory; auto-cleans after use."""
    if base_dir is None:
        base_dir = _settings.download_dir
    tmp_dir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ---------------------------
# Auth Helpers
# ---------------------------

async def is_user_authenticated(user_id: int, is_private: bool = True) -> bool:
    """Check if user is authenticated (Redis flag) or global admin. For groups, return True (group-level auth)."""
    if not is_private:
        return True  # Groups use /enablebot, no user auth
    if user_id in _settings.admins:
        return True  # Global admins bypass
    auth = await redis_get_json(f"user:{user_id}:auth")
    return bool(auth)

# ---------------------------
# User preferences (Redis)
# ---------------------------

DEFAULT_PREFS = {
    "max_size_mb": _settings.max_filesize_mb,      # user-specific override of upload limit
    "large_strategy": "ask",                        # ask | auto_downscale | transcode | split | audio | link
    "prefer_document": False,                       # send video as document to avoid Telegram compression
    "prefer_audio_as_voice": False                  # not used here, reserved for future
}

VALID_STRATEGIES = ["ask", "auto_downscale", "transcode", "split", "audio", "link"]


async def get_user_prefs(user_id: int) -> dict:
    key = f"prefs:{user_id}"
    prefs = await redis_get_json(key)
    if not prefs:
        prefs = DEFAULT_PREFS.copy()
        await redis_set_json(key, prefs)
    else:
        # ensure new keys exist if we update schema later
        for k, v in DEFAULT_PREFS.items():
            prefs.setdefault(k, v)
    return prefs


async def set_user_prefs(user_id: int, updates: dict):
    key = f"prefs:{user_id}"
    prefs = await get_user_prefs(user_id)
    prefs.update(updates)
    # clamp
    prefs["max_size_mb"] = max(50, min(1990, int(prefs["max_size_mb"])))
    strat = prefs.get("large_strategy", "ask")
    prefs["large_strategy"] = strat if strat in VALID_STRATEGIES else "ask"
    await redis_set_json(key, prefs)


def _prefs_kb(prefs: dict) -> InlineKeyboardMarkup:
    # Strategy row
    rows = []
    row = []
    for s in ["ask", "auto_downscale", "transcode"]:
        label = ("✅ " if prefs["large_strategy"] == s else "") + s.replace("_", " ").title()
        row.append(InlineKeyboardButton(label, callback_data=f"pref::strategy::{s}"))
    rows.append(row)
    row = []
    for s in ["split", "audio", "link"]:
        label = ("✅ " if prefs["large_strategy"] == s else "") + s.title()
        row.append(InlineKeyboardButton(label, callback_data=f"pref::strategy::{s}"))
    rows.append(row)

    # Size controls
    rows.append([
        InlineKeyboardButton("−200MB", callback_data="pref::maxsize::-200"),
        InlineKeyboardButton(f"Limit: {int(prefs['max_size_mb'])}MB", callback_data="noop"),
        InlineKeyboardButton("+200MB", callback_data="pref::maxsize::+200"),
    ])
    rows.append([
        InlineKeyboardButton("−50MB", callback_data="pref::maxsize::-50"),
        InlineKeyboardButton("+50MB", callback_data="pref::maxsize::+50"),
    ])

    # Toggles
    rows.append([
        InlineKeyboardButton(("✅ " if prefs["prefer_document"] else "") + "Send video as document",
                             callback_data="pref::toggle::prefer_document"),
    ])

    rows.append([InlineKeyboardButton("Close", callback_data="pref::close")])
    return InlineKeyboardMarkup(rows)


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    # Check auth for settings (optional; but consistent)
    if update.effective_chat.type == "private" and not await is_user_authenticated(user.id):
        await update.effective_message.reply_text("Authenticate with /auth <password> to access settings.")
        return
    prefs = await get_user_prefs(user.id)
    await update.effective_message.reply_text(
        "User settings ⚙️\nChoose your defaults below.",
        reply_markup=_prefs_kb(prefs)
    )


# ---------------------------
# Format selection UI
# ---------------------------

def _format_buttons(info: MediaInfo) -> InlineKeyboardMarkup:
    rows = []
    for fmt in info.formats:
        label = ("✅ " if fmt.cached else "") + fmt.note
        token = f"{fmt.format_id}"
        rows.append([InlineKeyboardButton(label, callback_data=f"dl::{token}")])
    rows.append([
        InlineKeyboardButton("Best", callback_data="dlbest"),
        InlineKeyboardButton("Audio", callback_data="dlaudio"),
        InlineKeyboardButton("Cancel", callback_data="dlcancel"),
    ])
    rows.append([
        InlineKeyboardButton("⚙️ Settings", callback_data="pref::open"),
    ])
    return InlineKeyboardMarkup(rows)


# ---------------------------
# Cache helpers
# ---------------------------

async def _get_cached_formats(url: str) -> dict[str, CacheItem]:
    uh = url_hash(url)
    async with SessionLocal() as ses:
        rows = (await ses.execute(select(CacheItem).where(CacheItem.url_hash == uh))).scalars().all()
        return {row.format_id: row for row in rows}


async def _send_from_cache(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str, chosen: FormatOption) -> bool:
    msg = update.effective_message
    cached = await _get_cached_formats(url)
    item = cached.get(chosen.format_id)
    if not item:
        return False
    caption = f"{item.title}\n[{item.provider}] {url}"
    try:
        if item.content_type == "video":
            await msg.reply_video(video=item.telegram_file_id, caption=caption, supports_streaming=True)
        elif item.content_type == "audio":
            await msg.reply_audio(audio=item.telegram_file_id, caption=caption)
        else:
            await msg.reply_document(document=item.telegram_file_id, caption=caption)
        return True
    except Exception as e:
        logger.warning("Failed to send from cache: {}", e)
        return False


# ---------------------------
# Logging
# ---------------------------

async def _log_job(user_id: int, chat_id: int, url: str, provider: str, status: str, started: float, error: str | None = None):
    dur_ms = int((time.time() - started) * 1000)
    async with SessionLocal() as ses:
        ses.add(JobLog(
            user_id=user_id,
            chat_id=chat_id,
            url=url,
            provider=provider,
            status=status,
            duration_ms=dur_ms,
            error=(error or "")[:5000]
        ))
        await ses.commit()


# ---------------------------
# Large-file helpers
# ---------------------------

def _bytes_limit_for(prefs: dict) -> int:
    return int(prefs.get("max_size_mb", _settings.max_filesize_mb)) * 1024 * 1024


async def _probe_format_size(url: str, format_id: str) -> Optional[int]:
    # lightweight probe via yt-dlp (no download)
    from yt_dlp import YoutubeDL
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "format": format_id,
    }
    if _settings.ytdlp_cookies_path:
        opts["cookiefile"] = _settings.ytdlp_cookies_path

    def _extract():
        with YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)

    try:
        info = await asyncio.to_thread(_extract)
        # When format is a single file, filesize may be info['filesize'] or 'filesize_approx'
        size = info.get("filesize") or info.get("filesize_approx")
        if size:
            return int(size)
        # If format is a combination (video+audio), check requested_formats
        reqs = info.get("requested_formats") or []
        total = 0
        found = False
        for r in reqs:
            fs = r.get("filesize") or r.get("filesize_approx")
            if fs:
                total += int(fs)
                found = True
        return total if found else None
    except Exception:
        return None


async def _extract_direct_url(url: str, format_id: str) -> Optional[str]:
    from yt_dlp import YoutubeDL
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "format": format_id,
    }
    if _settings.ytdlp_cookies_path:
        opts["cookiefile"] = _settings.ytdlp_cookies_path

    def _extract():
        with YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)

    try:
        info = await asyncio.to_thread(_extract)
        # Direct URL may be at info['url'], or in requested_formats
        if info.get("url"):
            return info["url"]
        reqs = info.get("requested_formats") or []
        # choose the last one (usually audio) isn't ideal. Prefer the largest stream (video)
        if reqs:
            # pick the first that has vcodec != none
            for r in reqs:
                if r.get("vcodec") and r.get("vcodec") != "none" and r.get("url"):
                    return r["url"]
            for r in reqs:
                if r.get("url"):
                    return r["url"]
        return None
    except Exception:
        return None


def _choose_smaller_format(info: MediaInfo, max_bytes: int, prefer_same_ext: Optional[str] = None) -> Optional[FormatOption]:
    # Choose best video format that has known filesize <= max_bytes
    candidates = []
    for f in info.formats:
        if f.content_type != "video":
            continue
        if f.filesize and f.filesize <= max_bytes:
            candidates.append(f)
    if not candidates:
        # fallback: audio if no video fits
        for f in info.formats:
            if f.content_type == "audio" and f.filesize and f.filesize <= max_bytes:
                candidates.append(f)
    if not candidates:
        return None
    # Sort candidates: highest height with smallest size close to limit; prefer matching ext first
    def score(x: FormatOption):
        size_score = x.filesize or 0
        height = 0
        for p in x.note.split():
            if p.endswith("p") and p[:-1].isdigit():
                height = int(p[:-1])
                break
        ext_bonus = 1 if prefer_same_ext and x.ext == prefer_same_ext else 0
        return (-ext_bonus, -height, -size_score)

    candidates.sort(key=score)
    return candidates[0]


async def _run_ffmpeg(cmd: list[str]) -> int:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        logger.error("ffmpeg error: {}", err.decode("utf-8", errors="ignore")[:2000])
    return proc.returncode


async def _transcode_to_target(input_path: Path, target_bytes: int, duration_sec: Optional[int]) -> Path:
    # Compute conservative bitrate to aim below target size
    # target_total_bps = (target_bytes * 8) / duration
    # reserve audio ~128kbps, set video bitrate accordingly, clamp
    if not duration_sec or duration_sec <= 0:
        # no duration; fallback to 720p CRF transcode
        out = input_path.with_suffix(".small.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf", "scale='min(1280,iw)':-2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out)
        ]
        rc = await _run_ffmpeg(cmd)
        if rc != 0:
            raise RuntimeError("Transcode failed")
        return out

    total_bps = max(400_000, int(target_bytes * 8 / duration_sec * 0.92))  # 8% safety margin
    audio_bps = 128_000
    video_bps = max(250_000, total_bps - audio_bps)
    # Limit resolution to 720p to keep bitrate effective
    out = input_path.with_suffix(".small.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", "scale='min(1280,iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-b:v", str(video_bps),
        "-maxrate", str(int(video_bps * 1.4)), "-bufsize", str(int(video_bps * 2.0)),
        "-c:a", "aac", "-b:a", str(audio_bps),
        "-movflags", "+faststart",
        str(out)
    ]
    rc = await _run_ffmpeg(cmd)
    if rc != 0:
        raise RuntimeError("Transcode failed")
    return out


async def _split_by_size(input_path: Path, target_bytes: int, duration_sec: Optional[int]) -> list[Path]:
    # Estimate bitrate from file size and duration (if available), then compute segment_time
    try:
        in_size = input_path.stat().st_size
    except Exception:
        in_size = target_bytes * 2
    if duration_sec and duration_sec > 0:
        bitrate_bps = int(in_size * 8 / duration_sec)
        seg_time = max(60, min(900, int(target_bytes * 8 / bitrate_bps * 0.92)))
    else:
        # without duration, pick a conservative 5 minutes
        seg_time = 300

    out_dir = input_path.parent / f"{input_path.stem}_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = out_dir / "part_%03d.mp4"

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-c", "copy", "-map", "0",
        "-f", "segment",
        "-segment_time", str(seg_time),
        "-reset_timestamps", "1",
        "-segment_format", "mp4",
        "-segment_format_options", "movflags=+faststart",
        str(out_pattern)
    ]
    rc = await _run_ffmpeg(cmd)
    if rc != 0:
        raise RuntimeError("Splitting failed")

    parts = sorted(out_dir.glob("part_*.mp4"))
    if not parts:
        raise RuntimeError("Splitting produced no parts")
    # Optional: further ensure each part < target_bytes; otherwise we could transcode too, but keep it simple.
    return parts


async def _extract_audio_file(input_path: Path) -> Path:
    out = input_path.with_suffix(".m4a")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-c:a", "aac", "-b:a", "192k",
        str(out)
    ]
    rc = await _run_ffmpeg(cmd)
    if rc != 0:
        raise RuntimeError("Audio extraction failed")
    return out


# ---------------------------
# Handlers: main flow
# ---------------------------

async def pick_medium_format(info: MediaInfo) -> Optional[FormatOption]:
    """Pick 'medium' quality: highest res <=720p, or closest, or any video, fallback to best."""
    video_formats = [f for f in info.formats if f.content_type == "video" and hasattr(f, 'height') and f.height]
    if video_formats:
        # Prefer <=720p, highest height
        candidates_720 = [f for f in video_formats if f.height <= 720]
        if candidates_720:
            return max(candidates_720, key=lambda f: f.height)
        # Closest to 720p if none <=720p
        def closeness(f):
            return abs(f.height - 720)
        return min(video_formats, key=closeness)
    
    # Broader fallback: Any video (even without height)
    any_video = next((f for f in info.formats if f.content_type == "video"), None)
    if any_video:
        return any_video
    
    # Last resort: Best overall (from _pick_best_format)
    return await _pick_best_format(info)


async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str, quick: Optional[str] = None, is_private: bool = True):
    """Handle URL download; is_private=True for PM checks."""
    logger.info(f"Caught and analyzing link: {url}")
    user = update.effective_user
    msg = update.effective_message

    # Auth check for private chats
    if is_private and not await is_user_authenticated(user.id):
        await msg.reply_text("🔒 Authenticate first with /auth <password> to download in PM.")
        return

    allowed, reason = await rate_limiter.allow_message(user.id)
    if not allowed:
        await msg.reply_text(reason)
        return

    started = time.time()
    try:
        cached_by_fmt = await _get_cached_formats(url)
        cached_format_ids = set(cached_by_fmt.keys())
        info = await analyze(url, cached_format_ids)
        

        # quick modes
        if quick in ("best", "audio", "medium"):
            if quick == "medium":
                chosen = await pick_medium_format(info)  # <-- Awaited
            else:
                chosen = await _pick_best_format(info, audio_only=(quick == "audio"))
            if not chosen:
                await msg.reply_text("No suitable formats found.")
                await _log_job(user.id, msg.chat_id, url, info.provider, "error", started, "No formats")
                return
            await _perform_download_send(update, context, url, info, chosen, started)
            return

        # Normal interactive format selection
        text = f"Found: {info.title}\nSource: {info.provider}\nPick a format:"
        await msg.reply_text(text, reply_markup=_format_buttons(info))
        # Keep in chat session
        context.chat_data["last_url"] = url
        context.chat_data["last_info"] = info
    except Exception as e:
        logger.exception("Analyze error")
        logger.error(f"Analyze error for {url}: {e}")
        await msg.reply_text(f"Failed to analyze link. {str(e)}")
        await _log_job(user.id, msg.chat_id, url, "unknown", "error", started, traceback.format_exc())


async def _pick_best_format(info: MediaInfo, audio_only: bool = False) -> Optional[FormatOption]:
    candidates = [f for f in info.formats if (f.content_type == "audio" if audio_only else f.content_type == "video")]
    if not candidates:
        candidates = info.formats
    return candidates[0] if candidates else None


@dataclass
class PostAction:
    # Optional post-processing after download when handling large files
    action: Optional[str] = None   # None | "transcode" | "split" | "audio" | "link"


def _large_file_options_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Auto downscale", callback_data="lf::downscale"),
            InlineKeyboardButton("Split", callback_data="lf::split"),
        ],
        [
            InlineKeyboardButton("Audio only", callback_data="lf::audio"),
            InlineKeyboardButton("Direct link", callback_data="lf::link"),
        ],
        [InlineKeyboardButton("Cancel", callback_data="lf::cancel")],
    ])


async def _send_direct_link(update: Update, url: str, chosen: FormatOption):
    msg = update.effective_message
    direct = await _extract_direct_url(url, chosen.format_id)
    if direct:
        await msg.reply_text(f"Direct media URL (may expire):\n{direct}")
    else:
        await msg.reply_text("Could not obtain a direct media URL for this format.")


async def _perform_download_send(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    url: str,
    info: MediaInfo,
    chosen: FormatOption,
    started: float,
    forced_action: Optional[str] = None  # None | "transcode" | "split" | "audio" | "link"
):
    user = update.effective_user
    msg = update.effective_message
    prefs = await get_user_prefs(user.id)
    max_bytes = _bytes_limit_for(prefs)

    # Serve from cache if possible (only original non-processed results)
    if not forced_action and await _send_from_cache(update, context, url, chosen):
        await _log_job(user.id, msg.chat_id, url, info.provider, "success", started)
        return

    # If no forced action, pre-check size and apply strategy
    if not forced_action:
        # 1) If chosen has known size > limit -> apply strategy
        est_size = chosen.filesize or await _probe_format_size(url, chosen.format_id)
        if est_size and est_size > max_bytes:
            strategy = prefs["large_strategy"]
            if strategy == "ask":
                q = await msg.reply_text(
                    f"Selected format is larger than your limit ({human_size(est_size)} > {human_size(max_bytes)}).\n"
                    f"Choose how to proceed:",
                    reply_markup=_large_file_options_kb()
                )
                # remember context for LF decision
                context.user_data["pending_lf"] = {
                    "url": url,
                    "format_id": chosen.format_id,
                    "info": info,
                    "message_id": q.message_id,
                    "chat_id": q.chat_id,
                }
                return
            elif strategy == "auto_downscale":
                alt = _choose_smaller_format(info, max_bytes, prefer_same_ext=chosen.ext)
                if alt:
                    await msg.reply_text(f"Selected auto downscale to: {alt.note}")
                    chosen = alt
                else:
                    # fallback to audio if nothing fits
                    aud = await _pick_best_format(info, audio_only=True)
                    if aud:
                        await msg.reply_text("No video fits your limit; falling back to best audio.")
                        chosen = aud
                    else:
                        await msg.reply_text("No format fits within your size limit.")
                        await _log_job(user.id, msg.chat_id, url, info.provider, "error", started, "No format fits limit")
                        return
            elif strategy in ("transcode", "split", "audio", "link"):
                forced_action = strategy
                if forced_action == "link":
                    await _send_direct_link(update, url, chosen)
                    await _log_job(user.id, msg.chat_id, url, info.provider, "success", started)
                    return

    # Acquire job slot
    allowed, reason = await rate_limiter.acquire_job(user.id)
    if not allowed:
        await msg.reply_text(reason)
        return

    progress_message = await msg.reply_text("Starting download...")

    def progress_hook(d: dict):
        try:
            if d.get("status") == "downloading":
                downloaded = d.get("downloaded_bytes") or 0
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                pct = int(downloaded * 100 / total) if total else 0
                eta = d.get("eta") or 0
                text = f"Downloading: {pct}% ({downloaded//1024//1024}MB/{(total or 0)//1024//1024}MB) ETA {eta}s"
                # Fire-and-forget: No await to avoid blocking
                asyncio.create_task(progress_message.edit_text(text))
            elif d.get("status") == "finished":
                asyncio.create_task(progress_message.edit_text("Processing..."))
        except Exception:
            pass

    try:
        # Wrap download + process + send in temp dir context
        with temp_download_dir() as tmp_dir:
            # Download the chosen item
            file_path, dl_info = await download(url, chosen, tmp_dir=tmp_dir, progress_cb=progress_hook)

            # Post-processing based on forced_action or post-check size
            final_paths: list[Path] = []
            content_type = chosen.content_type

            # If no forced action, verify actual size; if too large, apply fallback order
            size_bytes = file_path.stat().st_size
            if not forced_action and size_bytes > max_bytes:
                await progress_message.edit_text(
                    f"Downloaded file is {human_size(size_bytes)} which exceeds your limit ({human_size(max_bytes)})."
                )
                # Fallback chain by preference if not "ask"
                strategy = prefs["large_strategy"]
                if strategy == "ask":
                    # Ask user what to do now that we know it's too big
                    q = await msg.reply_text(
                        "How should I handle the large file?",
                        reply_markup=_large_file_options_kb()
                    )
                    context.user_data["pending_lf"] = {
                        "url": url,
                        "format_id": chosen.format_id,
                        "info": info,
                        "downloaded_path": str(file_path),
                        "message_id": q.message_id,
                        "chat_id": q.chat_id,
                    }
                    await progress_message.delete()
                    await rate_limiter.release_job(user.id)
                    return
                else:
                    forced_action = strategy

            # Apply any forced action (outputs stay in tmp_dir)
            if forced_action == "transcode":
                await progress_message.edit_text("Transcoding to fit your size limit...")
                out = await _transcode_to_target(file_path, _bytes_limit_for(prefs), info.duration)
                final_paths = [out]
                content_type = "video"
            elif forced_action == "split":
                await progress_message.edit_text("Splitting into parts to fit your size limit...")
                parts = await _split_by_size(file_path, _bytes_limit_for(prefs), info.duration)
                final_paths = parts
                content_type = "video"
            elif forced_action == "audio":
                await progress_message.edit_text("Extracting audio...")
                out = await _extract_audio_file(file_path)
                final_paths = [out]
                content_type = "audio"
            else:
                # Normal path: send the downloaded file
                final_paths = [file_path]

            caption = f"{info.title}\n[{info.provider}] {url}"
            sent_messages = []

            # Send files (possibly multiple parts)
            for idx, p in enumerate(final_paths, start=1):
                part_caption = caption if len(final_paths) == 1 else f"{caption}\nPart {idx}/{len(final_paths)}"
                with open(p, "rb") as f:
                    file_input = InputFile(f, filename=p.name)
                    if content_type == "video" and not (await get_user_prefs(user.id)).get("prefer_document"):
                        sent = await msg.reply_video(video=file_input, caption=part_caption, supports_streaming=True)
                    elif content_type == "audio":
                        sent = await msg.reply_audio(audio=file_input, caption=part_caption)
                    else:
                        sent = await msg.reply_document(document=file_input, caption=part_caption)
                    sent_messages.append(sent)
                # Extra clean (optional; context handles dir)
                p.unlink(missing_ok=True)

        await progress_message.delete()

        # Save to cache only when we didn't alter the file (no split/transcode/audio from video)
        if forced_action is None and len(final_paths) == 1:
            sent = sent_messages[0]
            try:
                async with SessionLocal() as ses:
                    uh = url_hash(url)
                    ses.add(CacheItem(
                        url_hash=uh,
                        source_url=url,
                        provider=info.provider,
                        title=info.title,
                        format_id=chosen.format_id,
                        content_type=chosen.content_type,
                        file_size=chosen.filesize or size_bytes,
                        telegram_file_id=(
                            sent.video.file_id if getattr(sent, "video", None) else
                            sent.audio.file_id if getattr(sent, "audio", None) else
                            sent.document.file_id
                        ),
                        extra={"ext": chosen.ext}
                    ))
                    await ses.commit()
            except Exception as e:
                logger.warning("Failed to save cache: {}", e)

        await _log_job(user.id, msg.chat_id, url, info.provider, "success", started)

    except Exception as e:
        logger.exception("Download/send failed")
        try:
            await progress_message.edit_text(f"Failed: {str(e)}")
        except Exception:
            pass
        await _log_job(user.id, msg.chat_id, url, info.provider, "error", started, traceback.format_exc())
    finally:
        await rate_limiter.release_job(user.id)


# ---------------------------
# Commands
# ---------------------------

async def dl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if update.effective_chat.type == "private" and not await is_user_authenticated(user.id):
        await msg.reply_text("🔒 Authenticate first with /auth <password> to use /dl in PM.")
        return
    args = context.args or []
    url = args[0] if args else None
    if not url:
        url_list = extract_urls(msg.text or "")
        if url_list:
            url = url_list[0]
    if not url:
        await msg.reply_text("Usage: /dl <url>")
        return
    is_private = update.effective_chat.type == "private"
    await handle_url(update, context, url, is_private=is_private)


async def best_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if update.effective_chat.type == "private" and not await is_user_authenticated(user.id):
        await msg.reply_text("🔒 Authenticate first with /auth <password> to use /best in PM.")
        return
    args = context.args or []
    if not args:
        await msg.reply_text("Usage: /best <url>")
        return
    is_private = update.effective_chat.type == "private"
    await handle_url(update, context, args[0], quick="best", is_private=is_private)


async def audio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if update.effective_chat.type == "private" and not await is_user_authenticated(user.id):
        await msg.reply_text("🔒 Authenticate first with /auth <password> to use /audio in PM.")
        return
    args = context.args or []
    if not args:
        await msg.reply_text("Usage: /audio <url>")
        return
    is_private = update.effective_chat.type == "private"
    await handle_url(update, context, args[0], quick="audio", is_private=is_private)


async def link_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.effective_message.text or ""
    urls = extract_urls(text)
    if not urls:
        return
    chat = update.effective_chat
    user = update.effective_user
    msg = update.effective_message
    is_private = chat.type == "private"

    # -------------------------
    # Group behavior
    # -------------------------
    if chat.type in ("group", "supergroup"):
        enabled = await redis_get_json(f"group:{chat.id}:enabled")
        if not enabled:
            # Bot is not enabled in this group by an admin
            return  # Silent ignore

        url = urls[0]
        started = time.time()  # <-- Define started here
        try:
            cached_by_fmt = await _get_cached_formats(url)
            cached_format_ids = set(cached_by_fmt.keys())
            info = await analyze(url, cached_format_ids)
            logger.info(f"Extracted {len(info.formats)} formats for {info.provider}: {[f.note for f in info.formats[:3]]}")  # Debug log
            chosen = await pick_medium_format(info)
            if not chosen:
                chosen = await _pick_best_format(info)
            if chosen:
                await _perform_download_send(update, context, url, info, chosen, started=time.time())
            else:
                await msg.reply_text("No video formats available (unsupported URL or audio-only content). Try /dl in PM for options.")
        except Exception as e:
            logger.error(f"Group download error: {e}")
            await msg.reply_text(f"Error analyzing link: {e}")
        return

    # -------------------------
    # Private chat behavior
    # -------------------------
    if is_private:
        if not await is_user_authenticated(user.id):
            await msg.reply_text("🔒 Authenticate first with /auth <password> to download links in PM.")
            return
        await handle_url(update, context, urls[0], is_private=is_private)

async def auth_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    if not args:
        await update.effective_message.reply_text("Usage: /auth <password>")
        return

    user = update.effective_user
    password = args[0]
    if password == _settings.admin_pass:
        auth_key = f"user:{user.id}:auth"
        existing = await redis_get_json(auth_key)
        if existing:
            await update.effective_message.reply_text("✅ Already authenticated! (Flag expires in 30 days.)")
        else:
            await redis_set_json(auth_key, True, ex=86400*30)
            await update.effective_message.reply_text("✅ Authentication successful! You can now download in PM and use inline. (Expires in 30 days.)")
    else:
        await update.effective_message.reply_text("❌ Wrong password.")

async def logout_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear user auth flag."""
    await delete_json(f"user:{update.effective_user.id}:auth")
    await update.effective_message.reply_text("🔓 Logged out. Run /auth <password> to re-authenticate.")


# ---------------------------
# Callback handler (formats, prefs, large-file choices)
# ---------------------------

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    data = query.data or ""
    await query.answer()

    # Preferences panel open
    if data == "pref::open":
        user = update.effective_user
        if update.effective_chat.type == "private" and not await is_user_authenticated(user.id):
            await query.edit_message_text("🔒 Authenticate first to access settings.")
            return
        prefs = await get_user_prefs(user.id)
        try:
            await query.edit_message_text("User settings ⚙️", reply_markup=_prefs_kb(prefs))
        except Exception:
            await query.message.reply_text("User settings ⚙️", reply_markup=_prefs_kb(prefs))
        return

    # Preferences updates
    if data.startswith("pref::"):
        user = update.effective_user
        if update.effective_chat.type == "private" and not await is_user_authenticated(user.id):
            await query.edit_message_text("🔒 Authenticate first to update settings.")
            return
        prefs = await get_user_prefs(user.id)
        parts = data.split("::")
        if len(parts) >= 3:
            action = parts[1]
            key_or_val = parts[2]
            try:
                if action == "strategy" and key_or_val in VALID_STRATEGIES:
                    await set_user_prefs(user.id, {"large_strategy": key_or_val})
                elif action == "maxsize":
                    delta = int(key_or_val)
                    await set_user_prefs(user.id, {"max_size_mb": int(prefs["max_size_mb"]) + delta})
                elif action == "toggle" and key_or_val in ["prefer_document", "prefer_audio_as_voice"]:
                    await set_user_prefs(user.id, {key_or_val: not prefs.get(key_or_val, False)})
                elif action == "close":
                    await query.edit_message_text("Closed settings.")
                    return
            except Exception as e:
                logger.error("Prefs update error: {}", e)
        prefs = await get_user_prefs(user.id)
        try:
            await query.edit_message_text("User settings ⚙️", reply_markup=_prefs_kb(prefs))
        except Exception:
            pass
        return

    # Large file decision
    if data.startswith("lf::"):
        choice = data.split("::", 1)[1]
        pend = context.user_data.get("pending_lf") or {}
        url = pend.get("url")
        fmt_id = pend.get("format_id")
        info = pend.get("info")
        downloaded_path = pend.get("downloaded_path")
        # Clean memory
        context.user_data["pending_lf"] = {}
        if not url or not fmt_id or not info:
            await query.edit_message_text("Session expired. Please send the link again.")
            return

        # Map choice to forced_action
        if choice == "cancel":
            await query.edit_message_text("Canceled.")
            return

        forced_action = None
        if choice == "downscale":
            # try choosing a smaller format before downloading
            prefs = await get_user_prefs(update.effective_user.id)
            alt = _choose_smaller_format(info, _bytes_limit_for(prefs))
            if not alt:
                await query.edit_message_text("No smaller video fits your size limit.")
                return
            await query.edit_message_text(f"Chosen: {alt.note}. Downloading...")
            await _perform_download_send(update, context, url, info, alt, started=time.time())
            return
        elif choice in ("split", "audio", "link"):
            forced_action = {"split": "split", "audio": "audio", "link": "link"}[choice]

        # If we already downloaded the big file (post-check path exists), reuse it for split/transcode/audio
        if downloaded_path and os.path.exists(downloaded_path) and forced_action in ("split", "audio", "transcode"):
            # Reuse by wrapping a minimal FormatOption (content_type video)
            faked = FormatOption(format_id=fmt_id, note="custom", filesize=None, content_type="video", ext="mp4", cached=False)
            await query.edit_message_text("Processing your choice...")
            await _perform_download_send(update, context, url, info, faked, started=time.time(), forced_action=forced_action)
            return
        else:
            # Normal path: perform with forced_action
            # Recreate chosen FormatOption from info
            chosen = next((f for f in info.formats if f.format_id == fmt_id), None)
            if not chosen:
                await query.edit_message_text("Format not available anymore.")
                return
            await query.edit_message_text("Processing your choice...")
            await _perform_download_send(update, context, url, info, chosen, started=time.time(), forced_action=forced_action)
            return

    # Download decisions (original logic)
    if data == "dlbest":
        url = context.chat_data.get("last_url")
        info = context.chat_data.get("last_info")
        if not url or not info:
            await query.edit_message_text("Session expired. Please send the link again.")
            return
        chosen = await _pick_best_format(info)
        if not chosen:
            await query.edit_message_text("No suitable format.")
            return
        await query.edit_message_text(f"Chosen: {chosen.note}. Downloading...")
        await _perform_download_send(update, context, url, info, chosen, started=time.time())
        return

    if data == "dlaudio":
        url = context.chat_data.get("last_url")
        info = context.chat_data.get("last_info")
        if not url or not info:
            await query.edit_message_text("Session expired. Please send the link again.")
            return
        chosen = await _pick_best_format(info, audio_only=True)
        if not chosen:
            await query.edit_message_text("No audio format found.")
            return
        await query.edit_message_text(f"Chosen: {chosen.note}. Downloading...")
        await _perform_download_send(update, context, url, info, chosen, started=time.time())
        return

    if data == "dlcancel":
        await query.edit_message_text("Canceled.")
        return

    if data.startswith("dl::"):
        url = context.chat_data.get("last_url")
        info = context.chat_data.get("last_info")
        if not url or not info:
            await query.edit_message_text("Session expired. Please send the link again.")
            return
        fid = data.split("::", 1)[1]
        chosen = next((f for f in info.formats if f.format_id == fid), None)
        if not chosen:
            await query.edit_message_text("Format not available.")
            return
        await query.edit_message_text(f"Chosen: {chosen.note}. Downloading...")
        await _perform_download_send(update, context, url, info, chosen, started=time.time())