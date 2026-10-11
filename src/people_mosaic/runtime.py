"""Locate resources inside the app, independently of the launcher's PATH."""

import shutil
import sys
from pathlib import Path

from .errors import UserError


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    if frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def tool_path(name: str) -> str:
    if name not in {"ffmpeg", "ffprobe"}:
        raise ValueError(f"Unknown bundled tool: {name}")
    if frozen():
        path = resource_root() / "tools" / name
        if path.is_file():
            return str(path)
        raise UserError("同梱の動画ツールが見つかりません。アプリを再インストールしてください。")
    path = shutil.which(name)
    if path:
        return path
    raise UserError("FFmpeg が必要です。brew install ffmpeg を実行してください。")
