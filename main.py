from __future__ import annotations

import base64
import json
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


def _decoded_image_dir() -> Path:
    decoded_dir = Path("./data/decoded_images").expanduser()
    decoded_dir.mkdir(parents=True, exist_ok=True)
    return decoded_dir


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
        files = _normalize_message(value[1])[0] # This looks wrong in original, let's fix
        files = _normalize_files(value[1])
        return text, files

    return str(value or "").strip(), []


def _merge_text_with_text_files(text: str, file_paths: list[str]) -> tuple[str, list[str]]:
    merged_text = text.strip()
    image_paths: list[str] = []

    text_suffixes = {".txt", ".md", ".json", ".csv", ".log", ".yaml", ".yml"}

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue

        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = (mime_type or "").lower()

        if mime_type.startswith("image/"):
            image_paths.append(file_path)
            continue

        is_text_like = mime_type.startswith("text/") or path.suffix.lower() in text_suffixes
        if not is_text_like:
            continue

        try:
            file_text = path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:  # noqa: BLE001
            continue

        if file_text:
            merged_text = f"{merged_text}\n{file_text}".strip() if merged_text else file_text

    return merged_text, image_paths


def _file_to_data_url(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"
    raw = Path(file_path).read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def encode_image_to_base64(image_file: Any) -> tuple[str, str]:
    files = _normalize_files(image_file)
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
    output_path = _decoded_image_dir() / f"decoded_{uuid4().hex}{ext}"
    output_path.write_bytes(raw)
    return str(output_path), str(output_path), f"Đã decode ảnh thành công: {output_path.name}"


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
        content.append({"path": image_path})

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
    text, image_paths = _merge_text_with_text_files(text, file_paths)
    if not text and not image_paths:
        raise gr.Error("Nhập nội dung hoặc tải ảnh lên trước khi gửi.")

    for image_path in image_paths:
        if not Path(image_path).exists():
            raise gr.Error(f"Không tìm thấy file ảnh: {image_path}")

    model = _resolve_model(model_name)
    client = _make_client(base_url, api_key)

    display_content = _build_display_content(text, image_paths)

    ui_history = list(ui_history) + [
        {"role": "user", "content": display_content},
        {"role": "assistant", "content": ""}
    ]

    user_content = _api_message_content(text, image_paths)
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
    api_history = list(api_history) + [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_text},
    ]
    yield ui_history, ui_history, api_history, _clear_input()


def clear_chat():
    return [], [], [], _clear_input()


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="ChatLLM Tester",
    ) as demo:
        gr.Markdown(
            "# ChatLLM Tester\n"
            "Giao diện chat để test model OpenAI-compatible, hỗ trợ nhiều ảnh upload hoặc paste."
        )

        with gr.Tabs():
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=650, label="Chat")
                        message = gr.MultimodalTextbox(
                            label="Nội dung",
                            placeholder="Nhập câu hỏi, rồi upload hoặc paste nhiều ảnh vào đây...",
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

            with gr.Tab("Base64 Ảnh"):
                gr.Markdown("Encode ảnh sang base64 hoặc decode base64 thành ảnh.")
                with gr.Row():
                    with gr.Column(scale=1):
                        encode_image_input = gr.Image(type="filepath", label="Ảnh cần encode")
                        encode_btn = gr.Button("Encode ảnh", variant="primary")
                        encoded_base64_output = gr.Textbox(label="Base64", lines=10)
                        copy_base64_btn = gr.Button("Copy Base64", variant="secondary")
                        encoded_data_url_output = gr.Textbox(label="Data URL", lines=10)
                        copy_data_url_btn = gr.Button("Copy Data URL", variant="secondary")

                    with gr.Column(scale=1):
                        decode_base64_input = gr.Textbox(label="Base64 hoặc Data URL", lines=10)
                        decode_btn = gr.Button("Decode base64", variant="primary")
                        decoded_image_output = gr.Image(type="filepath", label="Ảnh đã decode")
                        decoded_file_output = gr.File(label="Tải ảnh decode")
                        decode_status = gr.Markdown(value="")

                encode_btn.click(
                    encode_image_to_base64,
                    inputs=[encode_image_input],
                    outputs=[encoded_base64_output, encoded_data_url_output],
                )
                copy_base64_btn.click(
                    fn=None,
                    inputs=[encoded_base64_output],
                    js="""
                    (value) => {
                        navigator.clipboard.writeText(value || "");
                    }
                    """,
                )
                copy_data_url_btn.click(
                    fn=None,
                    inputs=[encoded_data_url_output],
                    js="""
                    (value) => {
                        navigator.clipboard.writeText(value || "");
                    }
                    """,
                )
                decode_btn.click(
                    decode_base64_to_image,
                    inputs=[decode_base64_input],
                    outputs=[decoded_image_output, decoded_file_output, decode_status],
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
        css="""
        footer {display: none !important;}
        #api-button {display: none !important;}
        """,
    )


if __name__ == "__main__":
    main()
