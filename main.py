from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

import gradio as gr
from openai import OpenAI


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM_PROMPT = (
    "Bạn là trợ lý test model. Trả lời ngắn gọn, chính xác, hỗ trợ ảnh khi cần."
)

_LOCAL_DOTENV_CACHE: dict[str, str] | None = None


def _load_local_dotenv() -> dict[str, str]:
    global _LOCAL_DOTENV_CACHE
    if _LOCAL_DOTENV_CACHE is not None:
        return _LOCAL_DOTENV_CACHE

    values: dict[str, str] = {}
    env_path = Path(__file__).with_name(".env")
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


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    return _load_local_dotenv().get(name, default)


def _env_any(names: list[str], default: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
        file_value = _load_local_dotenv().get(name, "").strip()
        if file_value:
            return file_value
    return default


def _normalize_files(value: Any) -> list[str]:
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


def _normalize_message(value: Any) -> tuple[str, list[str]]:
    if isinstance(value, dict):
        text = str(value.get("text", "") or "").strip()
        files = _normalize_files(value.get("files"))
        return text, files

    if isinstance(value, tuple) and len(value) == 2:
        text = str(value[0] or "").strip()
        files = _normalize_files(value[1])
        return text, files

    return str(value or "").strip(), []


def _split_multimodal_files(file_paths: list[str]) -> tuple[list[str], list[str]]:
    image_paths: list[str] = []
    text_chunks: list[str] = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            raise gr.Error(f"Không tìm thấy file: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = (mime_type or "").lower()

        if mime_type.startswith("image/"):
            image_paths.append(file_path)
            continue

        if mime_type.startswith("text/"):
            text_content = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text_content:
                text_chunks.append(text_content)
            continue

        raise gr.Error(
            f"File không hỗ trợ: {path.name}. Chỉ nhận ảnh hoặc text."
        )

    return image_paths, text_chunks


def _file_to_data_url(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"
    raw = Path(file_path).read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _build_user_content(text: str, image_paths: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _file_to_data_url(image_path)},
            }
        )
    return content or [{"type": "text", "text": " "}]


def _api_message_content(text: str, image_paths: list[str]) -> str | list[dict[str, Any]]:
    if image_paths:
        return _build_user_content(text, image_paths)
    return text or " "


def _resolve_model(model_name: str) -> str:
    return model_name.strip() or DEFAULT_MODEL


def _list_models(base_url: str, api_key: str) -> list[str]:
    client = _make_client(base_url, api_key)
    response = client.models.list()
    models = sorted({model.id for model in response.data if getattr(model, "id", None)})
    return models or [DEFAULT_MODEL]


def _load_builtin_locked_model(base_url: str, api_key: str):
    try:
        models = _list_models(base_url, api_key)
        selected = models[0]
        return (
            gr.update(choices=[selected], value=selected, interactive=False),
            f"Model có sẵn: đã load {len(models)} model từ /v1/models, dùng mặc định `{selected}`.",
        )
    except Exception as exc:  # noqa: BLE001
        return (
            gr.update(choices=[DEFAULT_MODEL], value=DEFAULT_MODEL, interactive=False),
            f"Model có sẵn: không tải được /v1/models ({exc}), dùng mặc định `{DEFAULT_MODEL}`.",
        )


def load_models(base_url: str, api_key: str, current_model: str | None = None):
    try:
        models = _list_models(base_url, api_key)
        selected = current_model if current_model in models else models[0]
        return gr.update(choices=models, value=selected), f"Đã tải {len(models)} model."
    except Exception as exc:  # noqa: BLE001
        fallback = current_model or DEFAULT_MODEL
        return gr.update(choices=[fallback], value=fallback), f"Không tải được model: {exc}"


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
        base = (custom_base_url or _env_any(["OPENAI_BASE_URL", "base_url"], "https://api.openai.com/v1")).strip()
        key = (custom_api_key or _env_any(["OPENAI_API_KEY", "api_key"], "")).strip()
        model = (custom_model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        return (
            gr.update(value=base, interactive=True),
            gr.update(value=key, interactive=True),
            gr.update(choices=[model], value=model, interactive=True),
            "Đã chuyển sang Custom. Giữ thông tin bạn đã nhập.",
        )

    builtin_base = _env_any(["OPENAI_BASE_URL", "base_url"], "https://api.openai.com/v1")
    builtin_key = _env_any(["OPENAI_API_KEY", "api_key"], "")
    model_update, status = _load_builtin_locked_model(builtin_base, builtin_key)
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
        model_update, status = _load_builtin_locked_model(base_url, api_key)
        return (
            model_update,
            status,
            custom_base_url,
            custom_api_key,
            custom_model,
        )

    try:
        models = _list_models(base_url, api_key)
        preferred = (current_model or custom_model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        selected = preferred if preferred in models else models[0]
        return (
            gr.update(choices=models, value=selected, interactive=True),
            f"Đã tải {len(models)} model.",
            base_url,
            api_key,
            selected,
        )
    except Exception as exc:  # noqa: BLE001
        fallback = (current_model or custom_model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        return (
            gr.update(choices=[fallback], value=fallback, interactive=True),
            f"Không tải được model: {exc}",
            base_url,
            api_key,
            fallback,
        )


def _make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=base_url.strip() or "https://api.openai.com/v1",
        api_key=api_key.strip() or _env("OPENAI_API_KEY", "EMPTY"),
    )


def _clear_input() -> dict[str, Any]:
    return {"text": "", "files": []}


def _build_display_content(text: str, image_paths: list[str]) -> str | list[Any]:
    if not image_paths:
        return text or "[Tin nhắn chỉ có ảnh]"

    content: list[Any] = []
    if text:
        content.append(text)

    for image_path in image_paths:
        content.append(
            gr.Image(
                value=image_path,
                show_label=False,
                interactive=False,
            )
        )

    return content


def chat(
    message: Any,
    ui_history: list[dict[str, Any]],
    api_history: list[dict[str, Any]],
    model_name: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
):
    text, file_paths = _normalize_message(message)
    image_paths, text_chunks = _split_multimodal_files(file_paths)

    merged_text = "\n\n".join(part for part in [text, *text_chunks] if part).strip()

    if not merged_text and not image_paths:
        raise gr.Error("Nhập nội dung hoặc tải ảnh lên trước khi gửi.")

    model = _resolve_model(model_name)
    client = _make_client(base_url, api_key)

    display_content = _build_display_content(merged_text, image_paths)

    ui_history = ui_history + [{"role": "user", "content": display_content}, {"role": "assistant", "content": ""}]

    user_content = _api_message_content(merged_text, image_paths)
    request_messages = [
        {"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT},
        *api_history,
        {"role": "user", "content": user_content},
    ]

    stream = client.chat.completions.create(
        model=model,
        messages=request_messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        stream=True,
    )

    assistant_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        assistant_text += delta

        ui_history[-1]["content"] = assistant_text or "..."
        yield ui_history, ui_history, api_history, _clear_input()

    assistant_text = assistant_text.strip() or "(không có nội dung trả về)"
    ui_history[-1]["content"] = assistant_text
    api_history = [
        *api_history,
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_text},
    ]
    yield ui_history, ui_history, api_history, _clear_input()


def clear_chat():
    return [], [], [], _clear_input()


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="ChatLLM Tester") as demo:
        gr.Markdown(
            "# ChatLLM Tester\n"
            "Giao diện chat để test model OpenAI-compatible, hỗ trợ nhiều ảnh upload hoặc paste."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=650, label="Chat")
                message = gr.MultimodalTextbox(
                    label="Nội dung",
                    placeholder="Nhập/paste text dài hoặc paste/upload ảnh vào đây...",
                    file_count="multiple",
                    file_types=None,
                )
                with gr.Row():
                    send_btn = gr.Button("Gửi", variant="primary")
                    clear_btn = gr.Button("Xoá chat")

            with gr.Column(scale=2):
                gr.Markdown("## Cấu hình")
                model_source = gr.Radio(
                    choices=["Model có sẵn", "Custom"],
                    value="Model có sẵn",
                    label="Nguồn cấu hình",
                )
                model_name = gr.Dropdown(
                    choices=[DEFAULT_MODEL],
                    value=DEFAULT_MODEL,
                    label="Model",
                    allow_custom_value=True,
                    interactive=False,
                )
                base_url = gr.Textbox(
                    label="Base URL",
                    value=_env_any(["OPENAI_BASE_URL", "base_url"], "https://api.openai.com/v1"),
                    placeholder="https://api.openai.com/v1",
                    interactive=False,
                )
                api_key = gr.Textbox(
                    label="API key",
                    value=_env_any(["OPENAI_API_KEY", "api_key"], ""),
                    placeholder="Nhập API key",
                    type="password",
                    interactive=False,
                )
                with gr.Row():
                    refresh_models_btn = gr.Button("Tải model", variant="secondary")
                model_status = gr.Markdown(value="")
                temperature = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=0.2,
                    step=0.1,
                    label="Temperature",
                )
                max_tokens = gr.Slider(
                    minimum=16,
                    maximum=8192,
                    value=1024,
                    step=16,
                    label="Max tokens",
                )
                system_prompt = gr.Textbox(
                    label="System prompt",
                    value=DEFAULT_SYSTEM_PROMPT,
                    lines=8,
                )
                gr.Markdown(
                    "**Gợi ý:** đổi Base URL và API key, sau đó tải model từ `/v1/models`."
                )

        ui_state = gr.State([])
        api_state = gr.State([])
        custom_base_url_state = gr.State(_env_any(["OPENAI_BASE_URL", "base_url"], "https://api.openai.com/v1"))
        custom_api_key_state = gr.State(_env_any(["OPENAI_API_KEY", "api_key"], ""))
        custom_model_state = gr.State(DEFAULT_MODEL)

        send_inputs = [
            message,
            ui_state,
            api_state,
            model_name,
            base_url,
            api_key,
            temperature,
            max_tokens,
            system_prompt,
        ]
        send_outputs = [chatbot, ui_state, api_state, message]

        send_btn.click(chat, inputs=send_inputs, outputs=send_outputs)
        message.submit(chat, inputs=send_inputs, outputs=send_outputs)

        base_url.change(
            remember_custom_inputs,
            inputs=[
                model_source,
                base_url,
                api_key,
                model_name,
                custom_base_url_state,
                custom_api_key_state,
                custom_model_state,
            ],
            outputs=[custom_base_url_state, custom_api_key_state, custom_model_state],
        )
        api_key.change(
            remember_custom_inputs,
            inputs=[
                model_source,
                base_url,
                api_key,
                model_name,
                custom_base_url_state,
                custom_api_key_state,
                custom_model_state,
            ],
            outputs=[custom_base_url_state, custom_api_key_state, custom_model_state],
        )
        model_name.change(
            remember_custom_inputs,
            inputs=[
                model_source,
                base_url,
                api_key,
                model_name,
                custom_base_url_state,
                custom_api_key_state,
                custom_model_state,
            ],
            outputs=[custom_base_url_state, custom_api_key_state, custom_model_state],
        )

        model_source.change(
            switch_model_source,
            inputs=[model_source, custom_base_url_state, custom_api_key_state, custom_model_state],
            outputs=[base_url, api_key, model_name, model_status],
        )

        refresh_models_btn.click(
            refresh_models_for_custom,
            inputs=[
                model_source,
                base_url,
                api_key,
                model_name,
                custom_base_url_state,
                custom_api_key_state,
                custom_model_state,
            ],
            outputs=[model_name, model_status, custom_base_url_state, custom_api_key_state, custom_model_state],
        )

        demo.load(
            switch_model_source,
            inputs=[model_source, custom_base_url_state, custom_api_key_state, custom_model_state],
            outputs=[base_url, api_key, model_name, model_status],
        )
        clear_btn.click(clear_chat, outputs=[chatbot, ui_state, api_state, message])

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(
        server_name=_env("HOST", "0.0.0.0"),
        server_port=int(_env("PORT", "7860")),
        share=bool(int(_env("SHARE", "0"))),
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
