"""
VAIO — Video Auto Intelligence Operator
---------------------------------------
Modular CLI system for:
  1. Audio extraction (FFmpeg)
  2. Caption generation (Whisper)
  3. SEO title + description generation
  4. Multilingual translation
  5. Caption translation
"""

__version__ = "1.0.0"
__author__ = "AXID.ONE - Number16BusShelter"
__license__ = "MIT"

# Optional: expose public modules for direct imports
from . import core

# Simple helper for runtime version access
def get_version() -> str:
    return __version__