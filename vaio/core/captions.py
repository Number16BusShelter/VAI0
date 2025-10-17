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
    META_FILENAME,
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
    "4) Limit the line length to 100-200 characters\n"
)

USER_PROMPT_CAPTION_TRANSLATE = (
    "Translate the following SRT subtitles from {src_lang} to {tgt_lang}. "
    "Keep timestamps, numbering, and layout identical.\n\n"
    "----- BEGIN SRT -----\n{content}\n----- END SRT -----"
)


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
    Writes outputs to captions/<video>.<tgt_lang>.srt.
    """
    captions_dir = video_path.parent / "captions"
    if not captions_dir.exists():
        print(f"❌ No 'captions' directory found next to {video_path.name}.")
        print(f"   Run `vaio audio \"{video_path.name}\"` first to generate captions.")
        sys.exit(1)

    # Find source SRT: captions/<stem>.<xx>.srt (e.g., .ru.srt, .en.srt)
    stem = video_path.stem
    candidates = sorted(captions_dir.glob(f"{stem}.*.srt"))

    if not candidates:
        print(f"❌ No captions found for {video_path.name} in '{captions_dir}'.")
        print(f"   Expected pattern: captions/{stem}.<lang>.srt  (e.g., {stem}.ru.srt)")
        sys.exit(1)

    # Prefer file that matches SOURCE_LANGUAGE if possible, else take the first
    pref_code = (SOURCE_LANGUAGE or "").lower()[:2]
    base_srt = None
    for c in candidates:
        # infer language code from filename suffixes: <stem>.<code>.srt
        # Path.suffixes example: ['.ru', '.srt'] → code = 'ru'
        try:
            code_guess = c.suffixes[-2].lstrip(".")
        except Exception:
            code_guess = ""
        if code_guess == pref_code:
            base_srt = c
            break
    if base_srt is None:
        base_srt = candidates[0]
        try:
            pref_code = base_srt.suffixes[-2].lstrip(".")
        except Exception:
            pref_code = "ru"  # fallback display only

    base_content = read_text(base_srt)
    print(f"🌍 Translating captions ({base_srt.name}) "
          f"from {SOURCE_LANGUAGE} → {', '.join(TARGET_LANGUAGES.values())}")

    results = {}
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {}
        for code, lang in TARGET_LANGUAGES.items():
            out_file = captions_dir / f"{stem}.{code}.srt"
            futures[ex.submit(
                translate_one,
                base_content,
                SOURCE_LANGUAGE,
                code,
                lang,
                out_file
            )] = (code, lang, out_file)

        for fut in cf.as_completed(futures):
            code, lang, _ = futures[fut]
            results[code] = fut.result()

    # Update metadata
    meta = load_meta(video_path)
    meta["stage"] = "captions_translated"
    meta["caption_source_lang"] = pref_code
    meta["caption_translations"] = {
        code: ("done" if ok else "failed") for code, ok in results.items()
    }
    save_meta(video_path, meta)

    print("\n✅ Caption translation complete.")
    print(f"Translated files saved under: {captions_dir}")
    return results

