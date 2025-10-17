#!/usr/bin/env python3
"""
vaio/core/captions.py
=====================
Stage 4 – Caption translation workflow.

Translates an existing `.srt` subtitle file into multiple languages while keeping
timecodes and structure intact.

Usage:
  vaio captions <video.mp4>
"""

from __future__ import annotations
import concurrent.futures as cf
import sys
import time
from pathlib import Path
import ollama

from .constants import (
    SOURCE_LANGUAGE,
    TARGET_LANGUAGES,
    OLLAMA_MODEL,
    TEMPERATURE,
    MAX_RETRIES,
    INITIAL_BACKOFF_S,
    CONCURRENCY,
)
from .utils import read_text, write_text, load_meta, save_meta


# ────────────────────────────────
# 🧠 PROMPTS
# ────────────────────────────────
SYSTEM_PROMPT_CAPTION_TRANSLATE = (
    "You are a professional subtitle translator. "
    "Translate all text lines between timestamps faithfully and naturally. "
    "Keep numbering and timecodes identical.\n"
    "Rules:\n"
    "1) Do not alter timestamps or formatting.\n"
    "2) Translate spoken lines only.\n"
    "3) Output ONLY the valid .srt file text.\n"
)

USER_PROMPT_CAPTION_TRANSLATE = (
    "Translate the following SRT subtitles from {src_lang} to {tgt_lang}. "
    "Keep timestamps, numbering, and layout identical.\n\n"
    "----- BEGIN SRT -----\n{content}\n----- END SRT -----"
)

# ────────────────────────────────
# 🧹 SANITIZATION HELPERS
# ────────────────────────────────
def clean_srt(text: str) -> str:
    """Remove hallucinations like 'Субтитры создавал DimaTorzok' and empty lines."""
    banned_phrases = [
        "субтитры создавал",
        "dimatorzok",
        "subtitles by",
        "edited by",
        "спасибо за просмотр",
        "thank you for watching",
    ]
    clean_lines = []
    for line in text.splitlines():
        if any(b in line.lower() for b in banned_phrases):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


# ────────────────────────────────
# 🔁 CORE CHAT LOGIC
# ────────────────────────────────
def chat_once(model: str, system_prompt: str, user_prompt: str) -> str:
    """Perform a single Ollama chat completion."""
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": TEMPERATURE},
        stream=False,
    )
    return (resp.get("message") or {}).get("content", "").strip()


def chat_with_retries(model: str, system_prompt: str, user_prompt: str) -> str:
    delay = INITIAL_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return chat_once(model, system_prompt, user_prompt)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"⚠️ Retry {attempt}/{MAX_RETRIES} after error: {e}")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Caption translation failed after all retries.")


# ────────────────────────────────
# 🌍 TRANSLATION WORKER
# ────────────────────────────────
def translate_one(content: str, src_lang: str, code: str, lang: str, out_path: Path):
    """Translate a single caption file."""
    user_prompt = USER_PROMPT_CAPTION_TRANSLATE.format(
        src_lang=src_lang, tgt_lang=lang, content=content
    )
    try:
        translated = chat_with_retries(OLLAMA_MODEL, SYSTEM_PROMPT_CAPTION_TRANSLATE, user_prompt)
        if not translated.strip():
            raise ValueError("Empty response")
        translated = clean_srt(translated)
        write_text(out_path, translated)
        print(f"✅ {lang} ({code}) → {out_path.name}")
        return True
    except Exception as e:
        print(f"❌ {lang} ({code}) error: {e}")
        return False


# ────────────────────────────────
# 🚀 MAIN ENTRY
# ────────────────────────────────
def process(video_path: Path):
    """
    Translate the existing captions/<video>.<src_lang>.srt file into all target languages.
    """
    captions_dir = video_path.parent / "captions"
    base_srt = captions_dir / f"{video_path.stem}.{}.srt"

    # Fallback to plain SRT if old naming still exists
    if not base_srt.exists():
        base_srt = video_path.with_suffix(".srt")

    if not base_srt.exists():
        print(f"❌ No SRT found for {video_path.name}. Run `vaio audio {video_path.name}` first.")
        sys.exit(1)

    base_content = read_text(base_srt)
    base_content = clean_srt(base_content)

    print(f"🌍 Translating captions from {SOURCE_LANGUAGE} → {', '.join(TARGET_LANGUAGES.values())}")

    results = {}
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {}
        for code, lang in TARGET_LANGUAGES.items():
            out_file = captions_dir / f"{video_path.stem}.{code}.srt"
            futures[ex.submit(translate_one, base_content, SOURCE_LANGUAGE, code, lang, out_file)] = (
                code,
                lang,
                out_file,
            )

        for fut in cf.as_completed(futures):
            code, lang, _ = futures[fut]
            results[code] = fut.result()

    meta = load_meta(video_path)
    meta["stage"] = "captions_translated"
    meta["caption_translations"] = {
        code: ("done" if ok else "failed") for code, ok in results.items()
    }
    save_meta(video_path, meta)

    print("\n✅ Caption translation complete.")
    print("Translated .srt files saved under 'captions/' next to the original video.")
    return results
