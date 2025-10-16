#!/usr/bin/env python3
"""
vaio/core/audio.py
==================
Handles:
 🎧 audio extraction via FFmpeg
 💬 caption generation via Whisper
 🧾 interactive caption verification

Works entirely in the same directory as the source video.
"""

from __future__ import annotations
import subprocess
import shutil
from pathlib import Path
import sys
import whisper

from .constants import (
    META_FILENAME,
    WHISPER_MODEL,
    SOURCE_LANGUAGE,
    DEFAULT_AUDIO_RATE,
    DEFAULT_AUDIO_CHANNELS,
)
from .utils import save_meta, load_meta, confirm


# ────────────────────────────────
# 🧹 TEXT SANITIZATION
# ────────────────────────────────
def clean_text(text: str) -> str:
    """Remove Whisper hallucinations like 'Субтитры создавал DimaTorzok' and noise."""
    banned_phrases = [
        "субтитры создавал",
        "dimatorzok",
        "subtitles by",
        "edited by",
        "спасибо за просмотр",
        "thank you for watching",
    ]
    for phrase in banned_phrases:
        if phrase.lower() in text.lower():
            return ""
    return text.strip()


# ────────────────────────────────
# 🎧 AUDIO EXTRACTION
# ────────────────────────────────
def extract_audio(video_path: Path) -> Path:
    audio_path = video_path.with_suffix(".mp3")

    print(f"🎧 Extracting audio → {audio_path.name}")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ar",
        str(DEFAULT_AUDIO_RATE),
        "-ac",
        str(DEFAULT_AUDIO_CHANNELS),
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("✅ Audio extracted.")
    meta = load_meta(video_path)
    meta["stage"] = "audio_done"
    save_meta(video_path, meta)
    return audio_path


# ────────────────────────────────
# 💬 CAPTION GENERATION
# ────────────────────────────────
def format_timestamp(seconds: float) -> str:
    """Format float seconds into hh:mm:ss,ms for SRT."""
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    hrs, s = divmod(s, 3600)
    mins, secs = divmod(s, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """Convert Whisper segments into valid SRT text, cleaning hallucinations."""
    lines = []
    index = 1
    for seg in segments:
        text = clean_text(seg["text"])
        if not text:
            continue
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        lines.append(f"{index}\n{start} --> {end}\n{text}\n")
        index += 1
    return "\n".join(lines)


def generate_captions(audio_path: Path, whisper_model: str = WHISPER_MODEL) -> Path:
    print(f"🧠 Generating captions using Whisper ({whisper_model})...")

    model = whisper.load_model(whisper_model)
    task = "translate" if SOURCE_LANGUAGE.lower() != "english" else "transcribe"
    result = model.transcribe(str(audio_path), task=task)

    captions_dir = audio_path.parent / "captions"
    captions_dir.mkdir(exist_ok=True)
    srt_path = captions_dir / f"{audio_path.stem}.ru.srt"

    srt_text = segments_to_srt(result["segments"])
    srt_path.write_text(srt_text, encoding="utf-8")
    print(f"✅ Captions saved → {srt_path}")

    # Automatically open in VS Code
    try:
        if shutil.which("code"):
            subprocess.Popen(["code", str(srt_path)])
            print("🧩 Opened captions in VS Code for review.")
        else:
            print("⚠️  VS Code not found in PATH. Skipping auto-open.")
    except Exception as e:
        print(f"⚠️  Could not auto-open in VS Code: {e}")

    meta = load_meta(audio_path)
    meta["stage"] = "captions_done"
    meta["detected_language"] = result.get("language", "unknown")
    save_meta(audio_path, meta)
    return srt_path


# ────────────────────────────────
# ✅ CAPTION VERIFICATION
# ────────────────────────────────
def verify(video_path: Path):
    """Ask user to verify and optionally edit captions manually."""
    srt_path = video_path.parent / "captions" / f"{video_path.stem}.ru.srt"
    if not srt_path.exists():
        print(f"❌ No captions found for {video_path.name}. Run `vaio audio {video_path.name}` first.")
        sys.exit(1)

    print(f"\n🧐 Captions ready for verification: {srt_path.name}")
    print("Open this file in your editor if needed, then confirm below.\n")

    if not confirm("Is caption file correct?"):
        print("❌ Aborted. Fix the SRT file and rerun `vaio continue` when ready.")
        sys.exit(0)

    confirmed_srt = srt_path.read_text(encoding="utf-8")
    print("✅ Captions confirmed by user.")

    meta = load_meta(video_path)
    meta["stage"] = "captions_verified"
    meta["verified_caption_length"] = len(confirmed_srt.splitlines())
    save_meta(video_path, meta)
    return srt_path


# ────────────────────────────────
# 🎯 MAIN ENTRY
# ────────────────────────────────
def process(video_path: Path):
    """Run extraction → caption generation → verification."""
    audio_path = extract_audio(video_path)
    srt_path = generate_captions(audio_path)
    verify(video_path)
    return srt_path
