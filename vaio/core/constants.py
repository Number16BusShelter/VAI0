"""
vaio/core/constants.py
======================
Central configuration constants for VAIO (Video Auto Intelligence Operator)
"""

from __future__ import annotations
import os

# ────────────────────────────────
# 📦 Project Info
# ────────────────────────────────
PROJECT_NAME = "VAIO"
PROJECT_VERSION = "1.0.1"
PROJECT_AUTHOR = "AXID.ONE"
PROJECT_LICENSE = "MIT"

# ────────────────────────────────
# 🌐 Language Configuration
# ────────────────────────────────
SOURCE_LANGUAGE = "English"
SOURCE_LANGUAGE_CODE = "en"

# Supported target translations (ISO codes → human-readable)
TARGET_LANGUAGES = {
    "ar": "Arabic",
    "ja": "Japanese",
    "zh": "Mandarin",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
    "fr": "French",
    # "en": "English",
    "ru": "Russian"
}

# ────────────────────────────────
# 🧠 Model Configuration
# ────────────────────────────────
# Whisper model used for transcription
WHISPER_MODEL = os.getenv("VAIO_WHISPER_MODEL", "large-v3-turbo")

# Ollama model used for LLM operations
OLLAMA_HOST= "http://localhost:11434"
OLLAMA_MODEL = os.getenv("VAIO_MODEL", "gpt-oss")

# Model behavior
TEMPERATURE = float(os.getenv("VAIO_TEMPERATURE", "0.3"))
MAX_RETRIES = int(os.getenv("VAIO_MAX_RETRIES", "3"))
INITIAL_BACKOFF_S = float(os.getenv("VAIO_BACKOFF_S", "1.0"))

# ────────────────────────────────
# ⚙️ Operational Settings
# ────────────────────────────────
CONCURRENCY = min(8, os.cpu_count() or 4)
DEFAULT_AUDIO_RATE = 44100
DEFAULT_AUDIO_CHANNELS = 2

# File naming / metadata
META_FILENAME = ".vaio.json"
TD_FILE_PREFIX = "td"     # previously "title_description"
TD_FILE_EXTENSION = ".txt"
CAPTION_EXTENSION = ".srt"

# ────────────────────────────────
# 📄 Prompts (shared across modules)
# ────────────────────────────────

# TD Generation Constants
TD_MAX_CAPTION_LENGTH = 2000
TD_MAX_TITLE_LENGTH = 100
TD_TITLE_TRUNCATE_LENGTH = 97
TD_MAX_HASHTAGS = 15
TD_HASHTAG_MIX = {
    'broad': (3, 4),
    'niche': (5, 7), 
    'specific': (2, 3)
}

# Template Processing
TEMPLATE_COMMENT_PREFIX = "--"
TEMPLATE_BLOCK_PATTERN = r"<!--\s*<(?P<name>[^>]+)>\s*-->(?P<content>.*?)<!--\s*</\1>\s*-->"

# Semantic Block Names
BLOCK_NAMES = {
    'INSTRUCTIONS': 'Instructions',
    'CONTEXT': 'Context', 
    'VIDEO_NAME': 'Video Name',
    'VIDEO_DESCRIPTION': 'Video Description',
    'HASH_TAGS': 'Hash tags'
}

# Guidance blocks to remove from final output
GUIDANCE_BLOCKS = {'Instructions', 'Context', 'Video Name'}

# Default fallbacks
DEFAULT_HASHTAGS = ""
DEFAULT_DESCRIPTION_GUIDE = "Create engaging, SEO-optimized description"

TMP_FILENAME="tdtmp.txt"

SYSTEM_PROMPT_TITLE = (
    "You are an expert in YouTube SEO, media marketing, and title optimization. "
    "Your task is to analyze captions (SRT) and produce a concise, catchy, "
    "and SEO-optimized title.\n\n"
    "Rules:\n"
    "1) The title must reflect the video's main content.\n"
    "2) Make it emotional, attractive, and relevant to search trends.\n"
    "3) Output ONLY the final title text — no explanations, no notes."
)

USER_PROMPT_TITLE = (
    "Analyze the following subtitles (SRT) and generate a compelling YouTube title in {src_lang}. "
    "Use emotional, descriptive, and SEO-rich phrasing based on the video's content.\n\n"
    "----- SRT CONTENT -----\n{content}\n----- END SRT -----"
)

SYSTEM_PROMPT_DESC = (
    "You are an expert in multilingual SEO, content marketing, and YouTube optimization. "
    "Generate a full video description in {src_lang} using the provided captions and template.\n"
    "Rules:\n"
    "1) The first paragraph (hook) must summarize and emotionally engage the viewer.\n"
    "2) Include relevant SEO keywords naturally.\n"
    "3) Keep structure and formatting consistent with the provided template.\n"
    "4) Insert generated text in place of <Hook and SEO optimized video description from captions>.\n"
    "5) Do not alter other template content.\n"
    "6) Output ONLY the completed description, no comments or explanations."
)

USER_PROMPT_DESC = (
    "Here is the caption (SRT) and a layout template. "
    "Write a full YouTube description following the template’s format.\n\n"
    "----- CAPTIONS (SRT) -----\n{captions}\n----- END CAPTIONS -----\n\n"
    "----- TEMPLATE -----\n{template}\n----- END TEMPLATE -----"
)

# ────────────────────────────────
# 🧩 CLI Emoji Theme (optional)
# ────────────────────────────────
CLI_ICONS = {
    "audio": "🎧",
    "captions": "💬",
    "desc": "📝",
    "translate": "🌐",
    "continue": "⏭️",
    "debug": "🧪",
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
}
