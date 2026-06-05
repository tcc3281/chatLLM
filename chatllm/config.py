from __future__ import annotations

import os
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "Bạn là trợ lý test model. Trả lời ngắn gọn, chính xác, hỗ trợ ảnh khi cần. "
    "Khi người dùng upload file, nội dung file sẽ được convert sang markdown và bọc trong tag <file_attachment>."
)

_LOCAL_DOTENV_CACHE: dict[str, str] | None = None


def load_local_dotenv() -> dict[str, str]:
    global _LOCAL_DOTENV_CACHE
    if _LOCAL_DOTENV_CACHE is not None:
        return _LOCAL_DOTENV_CACHE

    values: dict[str, str] = {}
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value

    _LOCAL_DOTENV_CACHE = values
    return values


def env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    return load_local_dotenv().get(name, default)


def env_any(names: list[str], default: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
        file_value = load_local_dotenv().get(name, "").strip()
        if file_value:
            return file_value
    return default
