# vaio/kb/cli.py
from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import click

from .query import build_kb_for_video, set_kb_dir_for_video, _resolve_kb_dir_for_video
from .paths import DEFAULT_KB_DIR, ensure_default_dirs
from .store import collection_stats, clear_index, debug_list_docs


@click.group(help="🧠 Knowledge Base management commands")
def kb() -> None:
    """Top-level KB command group."""


# ─────────────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────────────
@kb.command("build", help="Build Knowledge Base for a project video")
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--knowledge",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Override knowledge directory (full path)",
)
def build_cmd(video: str, knowledge: str | None) -> None:
    ensure_default_dirs()
    video_path = Path(video)
    kb_dir = Path(knowledge).resolve() if knowledge else None

    if kb_dir is None:
        click.echo("ℹ️  No --knowledge given; using project config or default.")
    
    result = build_kb_for_video(video_path, kb_dir)
    kb_identifier = knowledge or _resolve_kb_dir_for_video(video_path)

    if kb_identifier is None:
        click.secho("❌ KB disabled for this project.", fg="yellow")
        return

    stats = collection_stats(kb_identifier)
    click.secho(
        f"📊 KB collection={stats['collection']} | docs={stats['count']} | kb_name={stats['kb_name']}",
        fg="green",
    )

    click.echo("\n🔍 Listing current KB entries (truncated preview):")
    debug_list_docs(kb_identifier, limit=10)


# ─────────────────────────────────────────────────────────────
# LIST
# ─────────────────────────────────────────────────────────────
@kb.command("list", help="List stored KB documents")
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--knowledge",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Custom KB directory (default: from meta or knowledge/default)",
)
def list_cmd(video: str, knowledge: str | None) -> None:
    ensure_default_dirs()
    kb_identifier = knowledge or _resolve_kb_dir_for_video(Path(video))
    
    if not kb_identifier:
        click.secho("❌ KB disabled for this project.", fg="red")
        return

    click.echo(f"📚 Listing KB for '{kb_identifier}' ...")
    debug_list_docs(kb_identifier, limit=20)


# ─────────────────────────────────────────────────────────────
# SET
# ─────────────────────────────────────────────────────────────
@kb.command("set", help="Set or disable Knowledge Base directory for a project video")
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--knowledge",
    type=str,
    required=True,
    help="Full path to knowledge dir; use 'none' to disable",
)
def set_cmd(video: str, knowledge: str) -> None:
    ensure_default_dirs()
    video_path = Path(video)
    
    if knowledge.strip().lower() in {"none", "null"}:
        set_kb_dir_for_video(video_path, None)
        click.secho("✅ KB disabled for this project.", fg="yellow")
        return

    kb_dir = Path(knowledge).resolve()
    kb_dir.mkdir(parents=True, exist_ok=True)
    set_kb_dir_for_video(video_path, kb_dir)
    click.secho(f"✅ KB directory set to {kb_dir}", fg="green")


# ─────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────
@kb.command("stats", help="Show Knowledge Base stats for a project video")
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
def stats_cmd(video: str) -> None:
    ensure_default_dirs()
    kb_identifier = _resolve_kb_dir_for_video(Path(video))
    
    if kb_identifier is None:
        click.echo("ℹ️  KB disabled (knowledge=null).")
        return

    stats = collection_stats(kb_identifier)
    click.secho(
        f"📊 KB collection={stats['collection']} | docs={stats['count']} | kb_name={stats['kb_name']}",
        fg="cyan",
    )
    click.secho(f"📍 Persist path: {stats['persist_path']}", fg="cyan")


# ─────────────────────────────────────────────────────────────
# CLEAR
# ─────────────────────────────────────────────────────────────
@kb.command("clear", help="Clear KB index for a project video (keeps files)")
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
def clear_cmd(video: str) -> None:
    ensure_default_dirs()
    video_path = Path(video)
    kb_identifier = _resolve_kb_dir_for_video(video_path)
    
    if not kb_identifier:
        click.secho("❌ KB disabled or not configured.", fg="red")
        return

    clear_index(kb_identifier)
    click.secho(f"🧹 Cleared index for '{kb_identifier}'", fg="yellow")