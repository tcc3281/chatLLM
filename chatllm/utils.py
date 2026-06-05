from __future__ import annotations

import base64
import datetime
import json
import mimetypes
import re
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import gradio as gr

from .config import env
from .schemas import MediaType, MediaUrl

if TYPE_CHECKING:
    from markitdown import MarkItDown

# ---------------------------------------------------------------------------
# MarkItDown (thread-safe singleton)
# ---------------------------------------------------------------------------

_MARKITDOWN_CLIENT: "MarkItDown | None" = None
_MARKITDOWN_LOCK = threading.Lock()


def get_markitdown() -> "MarkItDown":
    global _MARKITDOWN_CLIENT
    if _MARKITDOWN_CLIENT is None:
        with _MARKITDOWN_LOCK:
            if _MARKITDOWN_CLIENT is None:
                from markitdown import MarkItDown

                _MARKITDOWN_CLIENT = MarkItDown()
    return _MARKITDOWN_CLIENT


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

def decoded_image_dir() -> Path:
    decoded_dir = Path(env("CHAT_HISTORY_DIR", "./data/decoded_images")).expanduser().resolve()
    decoded_dir.mkdir(parents=True, exist_ok=True)
    return decoded_dir


def chat_history_dir() -> Path:
    history_dir = Path(env("CHAT_HISTORY_DIR", "./data/chat_history")).expanduser().resolve()
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


# ---------------------------------------------------------------------------
# Chat history persistence
# ---------------------------------------------------------------------------

def save_chat_history(
    session_id: str,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
) -> None:
    history_file = chat_history_dir() / f"history_{datetime.date.today()}.jsonl"
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": session_id,
        "config": config,
        "messages": messages,
    }
    with history_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# File/message normalization
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(r"(https?://[^\s<>'\"]+|data:(?:image|video)/[^\s<>'\"]+)", re.IGNORECASE)
_MARKDOWN_LOCAL_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\((?P<path>(?:file://)?/[^\s)]+)\)")
_LOCAL_FILE_PATH_PATTERN = re.compile(r"(?P<prefix>^|[\s([])(?P<path>(?:file://)?/[^\s<>'\"]+)")
_INLINE_IMAGE_MARKER_PATTERN = re.compile(r"\s*\[Image\s*#\d+\]\s*", re.IGNORECASE)
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
_VIDEO_SUFFIXES = {".mp4", ".mpeg", ".mpg", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
_TEXT_FILE_SUFFIXES = {".txt", ".md", ".json", ".csv", ".log", ".yaml", ".yml"}
_MARKITDOWN_SUFFIXES = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm", ".zip",
    ".msg", ".eml", ".mp3", ".wav",
}


def _split_url_trailing_punctuation(url: str) -> tuple[str, str]:
    trailing = ""
    while url and url[-1] in _TRAILING_URL_PUNCTUATION:
        trailing = url[-1] + trailing
        url = url[:-1]
    return url, trailing


def _media_type_from_url(url: str) -> MediaType | None:
    lowered_url = url.lower()
    if lowered_url.startswith("data:image/"):
        return "image_url"
    if lowered_url.startswith("data:video/"):
        return "video_url"

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "image_url"
    if suffix in _VIDEO_SUFFIXES:
        return "video_url"
    return None


def extract_media_urls(text: str) -> tuple[str, list[MediaUrl]]:
    media_urls: list[MediaUrl] = []

    def replace_match(match: re.Match[str]) -> str:
        raw_url = match.group(0)
        url, trailing = _split_url_trailing_punctuation(raw_url)
        media_type = _media_type_from_url(url)
        if not media_type:
            return raw_url

        media_urls.append({"type": media_type, "url": url})
        return trailing

    cleaned_text = _URL_PATTERN.sub(replace_match, text).strip()
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text, media_urls


def _dedupe_paths(paths: list[str]) -> list[str]:
    unique_paths: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        unique_paths.append(path)
        seen.add(path)
    return unique_paths


def _is_image_path_like(path: str) -> bool:
    mime_type, _ = mimetypes.guess_type(path)
    return (mime_type or "").lower().startswith("image/")


def _is_existing_image_path(path: str) -> bool:
    return Path(path).exists() and _is_image_path_like(path)


def extract_embedded_image_paths(text: str) -> tuple[str, list[str]]:
    image_paths: list[str] = []

    def replace_markdown_image(match: re.Match[str]) -> str:
        raw_path = match.group("path")
        path, trailing = _split_url_trailing_punctuation(raw_path)
        local_path = path.removeprefix("file://")
        if not _is_image_path_like(local_path):
            return match.group(0)

        if _is_existing_image_path(local_path):
            image_paths.append(local_path)
        return trailing

    def replace_match(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        raw_path = match.group("path")
        path, trailing = _split_url_trailing_punctuation(raw_path)
        local_path = path.removeprefix("file://")
        if not _is_image_path_like(local_path):
            return match.group(0)

        if _is_existing_image_path(local_path):
            image_paths.append(local_path)
        return f"{prefix}{trailing}"

    cleaned_text = _MARKDOWN_LOCAL_IMAGE_PATTERN.sub(replace_markdown_image, text)
    cleaned_text = _LOCAL_FILE_PATH_PATTERN.sub(replace_match, cleaned_text).strip()
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text, image_paths


def clean_inline_image_markers(text: str) -> str:
    cleaned_text = _INLINE_IMAGE_MARKER_PATTERN.sub(" ", text).strip()
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text


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
        text = clean_inline_image_markers(str(value.get("text", "") or ""))
        files = normalize_files(value.get("files"))
        text, embedded_image_paths = extract_embedded_image_paths(text)
        return text, _dedupe_paths([*files, *embedded_image_paths])

    if isinstance(value, tuple) and len(value) == 2:
        text = clean_inline_image_markers(str(value[0] or ""))
        files = normalize_files(value[1])
        text, embedded_image_paths = extract_embedded_image_paths(text)
        return text, _dedupe_paths([*files, *embedded_image_paths])

    text = clean_inline_image_markers(str(value or ""))
    text, embedded_image_paths = extract_embedded_image_paths(text)
    return text, embedded_image_paths


def _merge_text_block(current_text: str, block: str) -> str:
    return f"{current_text}\n\n{block}".strip() if current_text else block


def _is_text_like_file(path: Path, mime_type: str) -> bool:
    return mime_type.startswith("text/") or path.suffix.lower() in _TEXT_FILE_SUFFIXES


def _read_text_file(path: Path, current_text: str) -> str:
    try:
        file_text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not file_text:
            return current_text

        print(f"[chatLLM] Loaded text file: {path.name}")
        if not current_text:
            return file_text
        return _merge_text_block(current_text, f"--- File: {path.name} ---\n{file_text}")
    except Exception as exc:
        print(f"[chatLLM] Error reading text file {path.name}: {exc}")
        return current_text


def _convert_file_to_markdown(path: Path, current_text: str) -> str:
    try:
        print(f"[chatLLM] Converting with MarkItDown: {path.name}...")
        md_result = get_markitdown().convert(str(path))
        if not md_result or not md_result.text_content:
            return current_text

        extracted = md_result.text_content.strip()
        if not extracted:
            print(f"[chatLLM] MarkItDown returned empty content for {path.name}")
            return current_text

        print(f"[chatLLM] Successfully converted {path.name} ({len(extracted)} chars)")
        return _merge_text_block(
            current_text,
            f"<file_attachment name=\"{path.name}\">\n{extracted}\n</file_attachment>",
        )
    except Exception as exc:
        print(f"[chatLLM] MarkItDown conversion error for {path.name}: {exc}")
        return current_text


# ---------------------------------------------------------------------------
# YouTube extraction
# ---------------------------------------------------------------------------

def get_youtube_info(url: str) -> str:
    import yt_dlp

    ydl_opts: dict[str, Any] = {
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


# ---------------------------------------------------------------------------
# File merging (text / MarkItDown / images)
# ---------------------------------------------------------------------------

def merge_text_with_text_files(text: str, file_paths: list[str]) -> tuple[str, list[str], list[str]]:
    merged_text = text.strip()
    image_paths: list[str] = []
    other_file_paths: list[str] = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue

        if str(file_path).startswith(("https://www.youtube.com", "https://youtu.be")):
            try:
                print(f"[chatLLM] Transcribing YouTube with yt-dlp: {file_path}...")
                extracted = get_youtube_info(str(file_path))
                if extracted:
                    youtube_block = (
                        f"<youtube_transcription url=\"{file_path}\">\n"
                        f"{extracted}\n"
                        f"</youtube_transcription>"
                    )
                    merged_text = _merge_text_block(merged_text, youtube_block) if merged_text else extracted
                    print(f"[chatLLM] Successfully extracted YouTube info")
                continue
            except Exception as exc:
                print(f"[chatLLM] YouTube extraction error: {exc}")
                continue

        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = (mime_type or "").lower()

        if mime_type.startswith("image/"):
            image_paths.append(file_path)
            continue

        other_file_paths.append(file_path)

        if _is_text_like_file(path, mime_type):
            merged_text = _read_text_file(path, merged_text)
            continue

        if path.suffix.lower() in _MARKITDOWN_SUFFIXES:
            merged_text = _convert_file_to_markdown(path, merged_text)

    return merged_text, image_paths, other_file_paths


# ---------------------------------------------------------------------------
# Data URL helpers
# ---------------------------------------------------------------------------

def file_to_data_url(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"
    raw = Path(file_path).read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


# ---------------------------------------------------------------------------
# Base64 encode / decode images
# ---------------------------------------------------------------------------

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
    except Exception as exc:
        raise gr.Error(f"Base64 không hợp lệ: {exc}") from exc

    if not mime_type.startswith("image/"):
        mime_type = "image/png"

    ext = mimetypes.guess_extension(mime_type) or ".png"
    output_path = decoded_image_dir() / f"decoded_{uuid.uuid4().hex}{ext}"
    output_path.write_bytes(raw)
    return str(output_path), str(output_path), f"Đã decode ảnh thành công: {output_path.name}"
