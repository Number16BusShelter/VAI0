"""
vaio/core/tts.py
================
Stage 5 — Text-to-Speech (TTS) Generation

Converts multilingual caption (.srt) files into MP3 narrations
using Kokoro-TTS (local, offline).


Usage:
  vaio tts <video.mp4>
"""

from __future__ import annotations
from pathlib import Path
import subprocess
import numpy as np
import torch
import soundfile as sf
from kokoro import KPipeline
from vaio.core.utils import read_text, load_meta, save_meta
from vaio.core.constants import TARGET_LANGUAGES


# ────────────────────────────────
# 🌍 Kokoro language mapping
# ────────────────────────────────
LANG_MAP = {
    "en": "a",  # American English
    "es": "e",  # Spanish
    "fr": "f",  # French
    "it": "i",  # Italian
    "pt": "p",  # Portuguese (Brazil)
    "hi": "h",  # Hindi
    "ja": "j",  # Japanese
    "zh": "z",  # Mandarin Chinese
}

UNSUPPORTED_LANGS = {"ar", "de", "ru"}  # Not supported by Kokoro


# ────────────────────────────────
# 🧩 Helpers
# ────────────────────────────────
def get_pipe(lang_code: str):
    """Return Kokoro voice pipeline for given language code."""
    if lang_code in UNSUPPORTED_LANGS:
        print(f"⚠️  Skipping unsupported language: {lang_code}")
        return None

    key = LANG_MAP.get(lang_code[:2])
    if not key:
        print(f"⚠️  Language {lang_code} not supported by Kokoro-TTS.")
        return None

    try:
        return KPipeline(lang_code=key, repo_id="hexgrad/Kokoro-82M")
    except Exception as e:
        print(f"❌  Failed to initialize Kokoro for {lang_code}: {e}")
        return None


def save_as_mp3(wav_path: Path) -> Path:
    """Convert .wav to .mp3 using FFmpeg."""
    mp3_path = wav_path.with_suffix(".mp3")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(wav_path),
                "-codec:a", "libmp3lame",
                "-qscale:a", "2",
                str(mp3_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        wav_path.unlink(missing_ok=True)
        return mp3_path
    except Exception as e:
        print(f"⚠️  Could not convert to MP3 ({e}). WAV file kept.")
        return wav_path


def extract_text_from_srt(srt_file: Path) -> str:
    """Remove timestamps and numbering from an .srt file."""
    srt_text = read_text(srt_file)
    lines = [
        line.strip()
        for line in srt_text.splitlines()
        if "-->" not in line and not line.strip().isdigit() and line.strip()
    ]
    return " ".join(lines)


def to_2d_np(wf) -> np.ndarray:
    """Normalize waveform to (samples, channels) float32 numpy array."""
    if isinstance(wf, torch.Tensor):
        wf = wf.detach().cpu()
        if wf.ndim == 0:
            wf = wf.unsqueeze(0)
        if wf.ndim == 1:
            wf = wf.unsqueeze(0)  # [1, N]
        wf = wf.transpose(0, 1).contiguous().numpy()
    else:
        wf = np.asarray(wf, dtype=np.float32)
        if wf.ndim == 1:
            wf = wf[:, None]
    return wf.astype(np.float32, copy=False)


def collect_audio_from_generator(gen):
    """
    Collect and concatenate audio from Kokoro generator.
    Supports both:
      - New format: out.audio -> torch.Tensor
      - Old format: out.audio -> {"waveform": tensor, "sample_rate": int}
    """
    chunks = []
    sr = None

    for out in gen:
        audio = getattr(out, "audio", None)

        # new format → Tensor
        if isinstance(audio, torch.Tensor):
            wf = audio
            sr = getattr(out, "sample_rate", sr or 22050)
        # old format → dict
        elif isinstance(audio, dict):
            wf = audio.get("waveform", None)
            sr = audio.get("sample_rate", sr or 22050)
        else:
            continue

        if wf is None or not torch.is_tensor(wf) or wf.numel() == 0:
            continue

        wf_np = to_2d_np(wf)
        if wf_np.size == 0:
            continue
        chunks.append(wf_np)

    if not chunks or sr is None:
        return None, None

    waveform = np.concatenate(chunks, axis=0)
    return waveform, int(sr)



# ────────────────────────────────
# 🎙️ Main process
# ────────────────────────────────
def process(video_path: Path):
    """Generate TTS MP3 narrations from translated captions (.srt files)."""
    captions_dir = video_path.parent / "captions"
    if not captions_dir.exists():
        print("❌  No captions folder found. Run `vaio captions` first.")
        return

    tts_dir = video_path.parent / "tts"
    tts_dir.mkdir(exist_ok=True)

    print(f"🎙️  Starting TTS generation from captions for '{video_path.name}'…")

    results: dict[str, bool] = {}

    for code, lang in TARGET_LANGUAGES.items():
        srt_file = captions_dir / f"{video_path.stem}.{code}.srt"
        if not srt_file.exists():
            print(f"⚠️  Skipping {lang} — no {srt_file.name} found.")
            results[code] = False
            continue

        pipe = get_pipe(code)
        if not pipe:
            results[code] = False
            continue

        try:
            text = extract_text_from_srt(srt_file)
            if not text.strip():
                print(f"⚠️  Empty text in {srt_file.name}, skipping.")
                results[code] = False
                continue

            gen = pipe(text, voice="af_heart", speed=1.0)
            waveform, sample_rate = collect_audio_from_generator(gen)

            if waveform is None or sample_rate is None:
                print(f"⚠️  No audio returned for {lang}.")
                results[code] = False
                continue

            wav_path = tts_dir / f"{video_path.stem}.{code}.wav"
            sf.write(wav_path, waveform, sample_rate)
            mp3_path = save_as_mp3(wav_path)

            print(f"✅  {lang} narration saved → {mp3_path.name}")
            results[code] = True

        except Exception as e:
            print(f"❌  {lang} ({code}) generation failed: {e}")
            results[code] = False

    # Update metadata
    meta = load_meta(video_path)
    meta["stage"] = "tts_done"
    meta.setdefault("tts_from_captions", {}).update(
        {k: "done" if v else "failed" for k, v in results.items()}
    )
    save_meta(video_path, meta)

    print("🎧  TTS generation completed for all supported captions.")
