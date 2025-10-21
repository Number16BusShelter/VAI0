# vaio/kb/__init__.py
from __future__ import annotations

"""
Knowledge Base package for VAIO.
Provides Click CLI group and supporting utilities.
"""

# Expose the Click command group so main CLI can import it
from .cli import kb

__all__ = ["kb"]
