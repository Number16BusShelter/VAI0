#!/usr/bin/env python3
"""
vaio/core/full_auto.py
======================
Full staged orchestrator:
 audio → captions → verify → TD → TD translate → SRT translate
"""

from __future__ import annotations
from pathlib import Path
from .utils import load_meta
from . import audio, description, translate, captions, tts


def run(video_path: Path, template_path: Path | None = None):
    """
    Execute the full pipeline, resuming from the last known stage in .vaio.json.
    Pauses where manual confirmation is required (captions + TD).
    """
    video_path = video_path.resolve()
    meta = load_meta(video_path)
    stage = meta.get("stage", "audio_pending")
    print(f"🧩 Full-auto: current stage: {stage}")

    # 1) Audio + Captions (if needed)
    if stage in ("init", "audio_pending", "audio_done"):
        print("🎧 Step 1: audio extraction + caption generation")
        audio_path = audio.extract_audio(video_path)
        srt_path = audio.generate_captions(audio_path)
        print(f"➡️  Captions ready at {srt_path}")

        print("🛑 Manual checkpoint: caption verification...")
        audio.verify(video_path)           # exits if user says No
        meta = load_meta(video_path)
        stage = meta.get("stage", "captions_verified")

    # 2) Generate TD (title+description)
    if stage in ("captions_verified", "captions_done"):
        print("📝 Step 2: generating TD (title + description)")
        description.process(video_path, template_path)   # asks for confirmation inside
        meta = load_meta(video_path)
        stage = meta.get("stage", "description_done")

    # 3) Translate TD to all languages
    if stage == "description_done":
        print("🌐 Step 3: translating TD into target languages")
        translate.process(video_path)
        meta = load_meta(video_path)
        stage = meta.get("stage", "translated")

    # 4) Translate Captions to all languages
    if stage == "translated":
        print("💬 Step 4: translating captions into target languages")
        captions.process(video_path)
        meta = load_meta(video_path)
        stage = meta.get("stage", "captions_translated")

    if stage == "captions_translated":
        print("🎉 Full-auto pipeline complete.")
        print("🎧 Step 5: Generating multilingual voiceovers...")
        tts.process(video_path)


    else:
        print(f"ℹ️  Stopped at stage: {stage} (run again to continue)")
