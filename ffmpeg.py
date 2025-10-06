from __future__ import annotations
import asyncio
import math
import os
from pathlib import Path
from typing import Optional, List
from logger import setup_logging

logger = setup_logging()

async def _run(cmd: list[str]) -> int:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        logger.error("ffmpeg command failed: {}\nstdout: {}\nstderr: {}", " ".join(cmd), out.decode(), err.decode())
    return proc.returncode

async def ffprobe_duration(path: Path) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        return None
    try:
        return float(out.decode().strip())
    except Exception:
        return None

async def transcode_to_target_size(input_path: Path, target_mb: int, duration_sec: Optional[float] = None) -> Path:
    # Estimate bitrate to fit within target size
    stat = input_path.stat()
    if duration_sec is None or duration_sec <= 0:
        duration_sec = await ffprobe_duration(input_path) or 0.0
    if duration_sec <= 0:
        # can't estimate; use a safe profile ~2Mbps
        video_bitrate_k = 2000
    else:
        target_bits = (target_mb - 10) * 1024 * 1024 * 8  # subtract overhead
        audio_bitrate_k = 128
        video_bitrate = max((target_bits / duration_sec) - (audio_bitrate_k * 1000), 300_000)
        video_bitrate_k = int(video_bitrate / 1000)

    output_path = input_path.with_suffix(".transcoded.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", "libx264", "-preset", "veryfast",
        "-b:v", f"{video_bitrate_k}k", "-maxrate", f"{int(video_bitrate_k*1.45)}k", "-bufsize", f"{int(video_bitrate_k*3)}k",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path)
    ]
    rc = await _run(cmd)
    if rc != 0 or not output_path.exists():
        raise RuntimeError("Transcode failed")
    return output_path

async def split_into_segments(input_path: Path, duration_sec: int) -> List[Path]:
    # Copy streams into segments of approximated duration
    pattern = input_path.with_suffix(".part_%03d.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c", "copy", "-map", "0",
        "-f", "segment", "-segment_time", str(duration_sec),
        "-reset_timestamps", "1",
        str(pattern)
    ]
    rc = await _run(cmd)
    if rc != 0:
        raise RuntimeError("Split failed")
    parts = sorted(input_path.parent.glob(pattern.name))
    if not parts:
        raise RuntimeError("No segments produced")
    return parts