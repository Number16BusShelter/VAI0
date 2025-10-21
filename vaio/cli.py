#!/usr/bin/env python3
from __future__ import annotations

"""
VAIO CLI
========
Video Auto Intelligence Operator — automated pipeline for:
 🎧 audio extraction
 💬 caption generation
 📝 SEO title & description creation
 🌐 translation & localization
"""

import os
import sys
import shutil
import platform
from pathlib import Path
from textwrap import dedent

import click

from vaio.core import audio, description, translate, captions, full_auto
from vaio.core.utils import load_meta, save_last_command
from vaio.core.constants import PROJECT_VERSION

# Import KB click group
from vaio.kb.cli import kb


# ────────────────────────────────
# Console banner
# ────────────────────────────────
def banner() -> None:
    click.secho(
        dedent(
            f"""
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃   VAIO v{PROJECT_VERSION} – Video Auto Intelligence   ┃
            ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
            """
        ),
        fg="cyan",
    )


# ────────────────────────────────
# Diagnostics
# ────────────────────────────────
def run_diagnostics() -> bool:
    """Check critical components and display status."""
    import requests

    def check_ffmpeg():
        return shutil.which("ffmpeg") is not None

    def check_whisper():
        try:
            import whisper  # noqa
            return True
        except Exception:
            return False

    def check_ollama():
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def check_permissions():
        test_file = Path.cwd() / ".vaio_test.tmp"
        try:
            test_file.write_text("ok")
            test_file.unlink()
            return True
        except Exception:
            return False

    checks = {
        "Python": sys.version.split()[0],
        "OS": platform.system(),
        "FFmpeg": "✅ OK" if check_ffmpeg() else "❌ Missing",
        "Whisper": "✅ OK" if check_whisper() else "❌ Missing",
        "Ollama": "✅ Running" if check_ollama() else "❌ Not reachable",
        "Write Access": "✅ OK" if check_permissions() else "❌ No permission",
    }

    click.echo()
    click.secho("🧪 VAIO System Diagnostics", bold=True, fg="cyan")
    for k, v in checks.items():
        click.echo(f" - {k}: {v}")

    critical = all([check_ffmpeg(), check_whisper(), check_ollama(), check_permissions()])
    click.echo()
    if critical:
        click.secho("✅ Environment ready! You can safely run VAIO.", fg="green")
    else:
        click.secho("⚠️ Some checks failed. Fix issues before running VAIO.", fg="yellow")

    return critical


# ────────────────────────────────
# Command group
# ────────────────────────────────
@click.group()
@click.version_option(PROJECT_VERSION, prog_name="VAIO")
def cli():
    """Video Auto Intelligence Operator (VAIO)"""
    banner()


# ────────────────────────────────
# Commands
# ────────────────────────────────
@cli.command("full-auto", help="🤖 Run the full pipeline end-to-end")
@click.argument("video_file", type=click.Path(exists=True))
@click.option("--template-file", type=click.Path(exists=True), default=None)
def full_auto_cmd(video_file, template_file):
    full_auto.run(Path(video_file).resolve(), Path(template_file).resolve() if template_file else None)
    save_last_command(Path(video_file), sys.argv)


@cli.command("audio", help="🎧 Extract audio and generate captions")
@click.argument("video_file", type=click.Path(exists=True))
def audio_cmd(video_file):
    audio.process(Path(video_file).resolve())
    save_last_command(Path(video_file), sys.argv)


@cli.command("desc", help="📝 Generate SEO title + description (TD)")
@click.argument("video_file", type=click.Path(exists=True))
@click.option("--template-file", type=click.Path(exists=True), default=None)
def desc_cmd(video_file, template_file):
    description.process(Path(video_file).resolve(), Path(template_file).resolve() if template_file else None)
    save_last_command(Path(video_file), sys.argv)


@cli.command("translate", help="🌐 Translate title & description into all languages")
@click.argument("video_file", type=click.Path(exists=True))
def translate_cmd(video_file):
    translate.process(Path(video_file).resolve())
    save_last_command(Path(video_file), sys.argv)


@cli.command("captions", help="💬 Translate captions into all languages")
@click.argument("video_file", type=click.Path(exists=True))
def captions_cmd(video_file):
    captions.process(Path(video_file).resolve())
    save_last_command(Path(video_file), sys.argv)


@cli.command("tts", help="🎙️ Generate voiceovers (Text-to-Speech) from captions")
@click.argument("video_file", type=click.Path(exists=True))
def tts_cmd(video_file):
    from vaio.core import tts
    tts.process(Path(video_file).resolve())
    save_last_command(Path(video_file), sys.argv)


@cli.command("continue", help="⏭️ Resume automatically from the last saved stage")
@click.argument("video_file", type=click.Path(exists=True))
def continue_cmd(video_file):
    video_path = Path(video_file).resolve()
    meta = load_meta(video_path)
    stage = meta.get("stage", "init")
    click.echo(f"Detected current stage: {stage}")

    if stage in ("init", "audio_pending", "audio_done"):
        click.echo("→ Running audio/caption verification...")
        audio.verify(video_path)
        stage = "captions_verified"

    if stage in ("captions_verified", "captions_done"):
        click.echo("→ Generating title & description...")
        description.process(video_path)
        stage = "description_done"

    if stage == "description_done":
        click.echo("→ Translating TD files...")
        translate.process(video_path)
        stage = "translated"

    if stage == "translated":
        click.echo("→ Translating captions...")
        captions.process(video_path)
        stage = "captions_translated"

    click.secho("🎉 All stages complete!", fg="green")


@cli.command("check", help="🧪 Run environment diagnostics")
def check_cmd():
    run_diagnostics()


# ────────────────────────────────
# KB integration
# ────────────────────────────────
cli.add_command(kb)


# ────────────────────────────────
# Entry point
# ────────────────────────────────
def main():
    # Auto mode: `vaio video.mp4`
    if len(sys.argv) == 2 and Path(sys.argv[1]).suffix.lower() in (".mp4", ".mov", ".mkv"):
        video_path = Path(sys.argv[1]).resolve()
        click.secho(f"🧩 Auto mode: Detected video file '{video_path.name}'", fg="cyan")
        return full_auto.run(video_path)

    cli()


if __name__ == "__main__":
    sys.exit(main())
