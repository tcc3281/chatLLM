from __future__ import annotations

import base64
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
from markitdown import MarkItDown

from .config import env

_MARKITDOWN_CLIENT: MarkItDown | None = None


def get_markitdown() -> MarkItDown:
    global _MARKITDOWN_CLIENT
    if _MARKITDOWN_CLIENT is None:
        _MARKITDOWN_CLIENT = MarkItDown()
    return _MARKITDOWN_CLIENT


def decoded_image_dir() -> Path:
    decoded_dir = Path("./data/decoded_images").expanduser()
    decoded_dir.mkdir(parents=True, exist_ok=True)
    return decoded_dir


def chat_history_dir() -> Path:
    history_dir = Path(env("CHAT_HISTORY_DIR", "./data/chat_history")).expanduser()
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def save_chat_history(
    session_id: str,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
):
    import datetime
    
    history_file = chat_history_dir() / f"history_{datetime.date.today()}.jsonl"
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": session_id,
        "config": config,
        "messages": messages,
    }
    with history_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def normalize_files(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, dict):
        value = value.get("files") or value.get("value") or value.get("data") or []

    if not isinstance(value, (list, tuple)):
        value = [value]

    paths: list[str] = []
    for item in value:
        path = None
        if isinstance(item, str):
            path = item
        elif isinstance(item, dict):
            path = item.get("path") or item.get("name")
        else:
            path = getattr(item, "path", None) or getattr(item, "name", None)

        if path:
            paths.append(str(path))

    return paths


def normalize_message(value: Any) -> tuple[str, list[str]]:
    if isinstance(value, dict):
        text = str(value.get("text", "") or "").strip()
        files = normalize_files(value.get("files"))
        return text, files

    if isinstance(value, tuple) and len(value) == 2:
        text = str(value[0] or "").strip()
        files = normalize_files(value[1])
        return text, files

    return str(value or "").strip(), []


def get_youtube_info(url: str) -> str:
    import yt_dlp
    
    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            title = info.get("title", "Unknown Title")
            description = info.get("description", "")
            channel = info.get("uploader", "Unknown Channel")
            duration = info.get("duration_string", "")
            
            output = f"# {title}\n"
            output += f"**Channel:** {channel} | **Duration:** {duration}\n\n"
            output += "## Description\n"
            output += description if description else "(No description available)"
            
            return output.strip()
    except Exception as exc:
        return f"Error extracting YouTube info: {exc}"


def merge_text_with_text_files(text: str, file_paths: list[str]) -> tuple[str, list[str], list[str]]:
    merged_text = text.strip()
    image_paths: list[str] = []
    other_file_paths: list[str] = []

    # Standard text-like suffixes
    text_suffixes = {".txt", ".md", ".json", ".csv", ".log", ".yaml", ".yml"}
    # Suffixes supported by MarkItDown [all]
    markitdown_suffixes = {
        ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm", ".zip",
        ".msg", ".eml", ".mp3", ".wav"
    }

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue

        # Handle YouTube URLs passed as "files"
        if str(file_path).startswith(("https://www.youtube.com", "https://youtu.be")):
            try:
                print(f"[chatLLM] Transcribing YouTube with yt-dlp: {file_path}...")
                extracted = get_youtube_info(str(file_path))
                if extracted:
                    merged_text = f"{merged_text}\n\n<youtube_transcription url=\"{file_path}\">\n{extracted}\n</youtube_transcription>".strip() if merged_text else extracted
                    print(f"[chatLLM] Successfully extracted YouTube info")
                continue
            except Exception as exc:
                print(f"[chatLLM] YouTube extraction error: {exc}")
                continue

        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = (mime_type or "").lower()

        # Handle Images
        if mime_type.startswith("image/"):
            image_paths.append(file_path)
            continue

        other_file_paths.append(file_path)

        # Handle plain text files
        is_text_like = mime_type.startswith("text/") or path.suffix.lower() in text_suffixes
        if is_text_like:
            try:
                file_text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if file_text:
                    merged_text = f"{merged_text}\n\n--- File: {path.name} ---\n{file_text}".strip() if merged_text else file_text
                    print(f"[chatLLM] Loaded text file: {path.name}")
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[chatLLM] Error reading text file {path.name}: {exc}")
                pass

        # Handle everything else with MarkItDown (PDF, Word, Excel, etc.)
        if path.suffix.lower() in markitdown_suffixes:
            try:
                print(f"[chatLLM] Converting with MarkItDown: {path.name}...")
                md_result = get_markitdown().convert(str(path))
                if md_result and md_result.text_content:
                    extracted = md_result.text_content.strip()
                    if extracted:
                        merged_text = f"{merged_text}\n\n<file_attachment name=\"{path.name}\">\n{extracted}\n</file_attachment>".strip() if merged_text else f"<file_attachment name=\"{path.name}\">\n{extracted}\n</file_attachment>"
                        print(f"[chatLLM] Successfully converted {path.name} ({len(extracted)} chars)")
                    else:
                        print(f"[chatLLM] MarkItDown returned empty content for {path.name}")
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[chatLLM] MarkItDown conversion error for {path.name}: {exc}")
                continue

    return merged_text, image_paths, other_file_paths


def file_to_data_url(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"
    raw = Path(file_path).read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def encode_image_to_base64(image_file: Any) -> tuple[str, str]:
    files = normalize_files(image_file)
    if not files:
        raise gr.Error("Vui lòng chọn ảnh trước khi encode.")

    image_path = files[0]
    path = Path(image_path)
    if not path.exists():
        raise gr.Error(f"Không tìm thấy file ảnh: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "application/octet-stream"
    if not mime_type.startswith("image/"):
        raise gr.Error("File đã chọn không phải ảnh.")

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"
    return encoded, data_url


def decode_base64_to_image(base64_value: str) -> tuple[str | None, str | None, str]:
    payload = (base64_value or "").strip()
    if not payload:
        raise gr.Error("Vui lòng nhập chuỗi base64 trước khi decode.")

    mime_type = "image/png"
    if payload.startswith("data:") and "," in payload:
        header, payload = payload.split(",", 1)
        if ";" in header:
            mime_type = header[5:].split(";", 1)[0] or "image/png"

    clean_payload = "".join(payload.split())
    try:
        raw = base64.b64decode(clean_payload, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Base64 không hợp lệ: {exc}") from exc

    if not mime_type.startswith("image/"):
        mime_type = "image/png"

    ext = mimetypes.guess_extension(mime_type) or ".png"
    output_path = decoded_image_dir() / f"decoded_{uuid.uuid4().hex}{ext}"
    output_path.write_bytes(raw)
    return str(output_path), str(output_path), f"Đã decode ảnh thành công: {output_path.name}"
