from __future__ import annotations

import uuid
from typing import Any

import gradio as gr

from .config import DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, env, env_any
from .llm import (
    chat,
    load_builtin_locked_model,
    list_models,
    remember_custom_inputs,
)
from .utils import (
    encode_image_to_base64,
    decode_base64_to_image,
)


def clear_chat():
    def _clear_input() -> dict[str, Any]:
        return {"text": "", "files": []}
    return [], [], [], _clear_input(), "**Token Usage:** _Chưa có dữ liệu_"


def switch_model_source(
    model_source: str,
    custom_base_url: str,
    custom_api_key: str,
    custom_model: str,
):
    if model_source == "Custom":
        base = (custom_base_url or env_any(["OPENAI_BASE_URL", "base_url"], "https://api.openai.com/v1")).strip()
        key = (custom_api_key or env_any(["OPENAI_API_KEY", "api_key"], "")).strip()
        model = (custom_model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        return (
            gr.update(value=base, interactive=True),
            gr.update(value=key, interactive=True),
            gr.update(choices=[model], value=model, interactive=True),
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
                            placeholder="Nhập câu hỏi, YouTube URL, rồi upload hoặc paste ảnh/file/audio...",
                            file_count="multiple",
                            file_types=["image", "audio", ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".txt", ".csv", ".json", ".md", ".msg", ".eml"],
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
                            value=env_any(["OPENAI_BASE_URL", "base_url"], "https://api.openai.com/v1"),
                            placeholder="https://api.openai.com/v1",
                            interactive=False,
                        )
                        api_key = gr.Textbox(
                            label="API key",
                            value=env_any(["OPENAI_API_KEY", "api_key"], ""),
                            placeholder="Nhập API key",
                            type="password",
                            interactive=False,
                        )
                        with gr.Row():
                            refresh_models_btn = gr.Button("Tải model", variant="secondary")
                            enable_thinking = gr.Checkbox(label="Enable Thinking", value=False)
                        model_status = gr.Markdown(value="")
                        usage_status = gr.Markdown(value="**Token Usage:** _Chưa có dữ liệu_")
                        temperature = gr.Slider(
                            minimum=0,
                            maximum=2,
                            value=0.2,
                            step=0.1,
                            label="Temperature",
                        )
                        top_p = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.9,
                            step=0.05,
                            label="Top-P",
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
        session_id_state = gr.State(lambda: uuid.uuid4().hex[:12])
        custom_base_url_state = gr.State(env_any(["OPENAI_BASE_URL", "base_url"], "https://api.openai.com/v1"))
        custom_api_key_state = gr.State(env_any(["OPENAI_API_KEY", "api_key"], ""))
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
            enable_thinking,
            top_p,
            session_id_state,
        ]
        send_outputs = [chatbot, ui_state, api_state, message, usage_status]

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
        clear_btn.click(clear_chat, outputs=[chatbot, ui_state, api_state, message, usage_status])

    return demo
