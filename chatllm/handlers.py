from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from .client import list_models
from .config import DEFAULT_SYSTEM_PROMPT, env_any
from .utils import get_markitdown, normalize_files


def clear_chat():
    def _clear_input() -> dict[str, Any]:
        return {"text": "", "files": []}

    return [], [], [], _clear_input(), "**Token Usage:** _Chưa có dữ liệu_"


def load_builtin_locked_model(base_url: str, api_key: str) -> tuple[gr.Dropdown, str]:
    try:
        models = list_models(base_url, api_key)
        selected = models[0]
        return (
            gr.update(choices=[selected], value=selected, interactive=False),
            f"Model có sẵn: đã load {len(models)} model từ /v1/models, chọn `{selected}`.",
        )
    except Exception as exc:
        return (
            gr.update(choices=[], value=None, interactive=False),
            f"Model có sẵn: không tải được /v1/models ({exc}). Kiểm tra Base URL/API key rồi tải lại.",
        )


def load_models(base_url: str, api_key: str, current_model: str | None = None) -> tuple[gr.Dropdown, str]:
    try:
        models = list_models(base_url, api_key)
        selected = current_model if current_model in models else models[0]
        return gr.update(choices=models, value=selected), f"Đã tải {len(models)} model."
    except Exception as exc:
        fallback = (current_model or "").strip()
        choices = [fallback] if fallback else []
        return gr.update(choices=choices, value=fallback or None), f"Không tải được model: {exc}"


def remember_custom_inputs(
    model_source: str,
    base_url: str,
    api_key: str,
    model_name: str,
    custom_base_url: str,
    custom_api_key: str,
    custom_model: str,
):
    if model_source != "Custom":
        return custom_base_url, custom_api_key, custom_model
    return base_url, api_key, model_name


def switch_model_source(
    model_source: str,
    custom_base_url: str,
    custom_api_key: str,
    custom_model: str,
):
    if model_source == "Custom":
        base = (custom_base_url or env_any(["OPENAI_BASE_URL", "base_url"], "https://api.openai.com/v1")).strip()
        key = (custom_api_key or env_any(["OPENAI_API_KEY", "api_key"], "")).strip()
        model = (custom_model or "").strip()
        return (
            gr.update(value=base, interactive=True),
            gr.update(value=key, interactive=True),
            gr.update(choices=[model] if model else [], value=model or None, interactive=True),
            "Đã chuyển sang Custom. Giữ thông tin bạn đã nhập.",
        )

    builtin_base = env_any(["OPENAI_BASE_URL", "base_url"], "https://api.openai.com/v1")
    builtin_key = env_any(["OPENAI_API_KEY", "api_key"], "")
    model_update, status = load_builtin_locked_model(builtin_base, builtin_key)
    return (
        gr.update(value=builtin_base, interactive=False),
        gr.update(value=builtin_key, interactive=False),
        model_update,
        status,
    )


def refresh_models_for_custom(
    model_source: str,
    base_url: str,
    api_key: str,
    current_model: str,
    custom_base_url: str,
    custom_api_key: str,
    custom_model: str,
):
    if model_source != "Custom":
        model_update, status = load_builtin_locked_model(base_url, api_key)
        return (
            model_update,
            status,
            custom_base_url,
            custom_api_key,
            custom_model,
        )

    try:
        models = list_models(base_url, api_key)
        preferred = (current_model or custom_model or "").strip()
        selected = preferred if preferred in models else models[0]
        return (
            gr.update(choices=models, value=selected, interactive=True),
            f"Đã tải {len(models)} model.",
            base_url,
            api_key,
            selected,
        )
    except Exception as exc:
        fallback = (current_model or custom_model or "").strip()
        return (
            gr.update(choices=[fallback] if fallback else [], value=fallback or None, interactive=True),
            f"Không tải được model: {exc}",
            base_url,
            api_key,
            fallback,
        )


def convert_to_markdown(
    file: str | list | None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str | None, str]:
    files = normalize_files(file)
    if not files:
        return None, "Vui lòng chọn file trước khi convert."

    path = Path(files[0])
    if not path.exists():
        return None, f"Không tìm thấy file: {path}"

    try:
        progress(0, desc="Đang đọc file...")
        print(f"[chatLLM] Converting with MarkItDown: {path.name}...")
        result = get_markitdown().convert(str(path))
        progress(0.8, desc="Đang xử lý...")
        if result and result.text_content:
            markdown = result.text_content.strip()
            print(f"[chatLLM] Successfully converted {path.name} ({len(markdown)} chars)")
            progress(1.0, desc="Hoàn tất")
            return markdown, f"Convert thành công: {path.name} ({len(markdown)} ký tự)"
        return None, f"Không thể convert {path.name}: nội dung rỗng."
    except Exception as exc:
        print(f"[chatLLM] MarkItDown conversion error for {path.name}: {exc}")
        return None, f"Lỗi convert {path.name}: {exc}"
