# vaio/core/utils.py
from __future__ import annotations
import json
import sys
import time
import concurrent.futures as cf
from datetime import datetime
from pathlib import Path
import ollama

from .constants import (
    OLLAMA_MODEL,
    TEMPERATURE,
    MAX_RETRIES,
    INITIAL_BACKOFF_S,
)

# ---------- Text I/O ----------
def read_text(path: Path) -> str:
    """Safely read text files with UTF-8 or UTF-16 fallback."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-16").strip()


def write_text(path: Path, text: str):
    """Write text to file with UTF-8 encoding, ensuring trailing newline."""
    path.write_text(text.strip() + "\n", encoding="utf-8")


# ---------- Confirmation ----------
def confirm(prompt: str) -> bool:
    """Ask user for yes/no confirmation."""
    answer = input(f"{prompt} (Y/n): ").strip().lower()
    return answer in {"y", "yes", ""}


# ---------- Metadata (.vaio.json) ----------
def _meta_path(video_path: Path) -> Path:
    """Return metadata file path next to the video."""
    return video_path.with_suffix(".vaio.json")


def load_meta(video_path: Path) -> dict:
    """Load or initialize metadata for a video."""
    meta_file = _meta_path(video_path)
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ Failed to read metadata: {e}")
    return {"file": str(video_path), "stage": "audio_pending"}


def save_meta(video_path: Path, meta: dict):
    """Persist metadata to disk."""
    meta["last_updated"] = datetime.utcnow().isoformat() + "Z"
    meta_file = _meta_path(video_path)
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------- Ollama Chat Wrappers ----------
def retry(func, *args, **kwargs):
    """Generic exponential backoff retry wrapper."""
    delay = INITIAL_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"⚠️ Retry {attempt}/{MAX_RETRIES} after error: {e}")
            time.sleep(delay)
            delay *= 2


def chat_once(model: str, system_prompt: str, user_prompt: str) -> str:
    """Single Ollama chat call."""
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": TEMPERATURE},
        stream=False,
    )
    return (resp.get("message") or {}).get("content", "").strip()


def chat_with_retries(model: str, system_prompt: str, user_prompt: str) -> str:
    """Ollama chat call with retry and backoff."""
    return retry(chat_once, model, system_prompt, user_prompt)

# ---------- Filesystem ----------
def ensure_dir(path: Path):
    """Ensure that the directory exists (mkdir -p)."""
    path.mkdir(parents=True, exist_ok=True)
    return path
