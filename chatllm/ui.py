from __future__ import annotations

import uuid

import gradio as gr

from .chat import chat
from .config import DEFAULT_SYSTEM_PROMPT, env_any
from .handlers import (
    clear_chat,
    convert_to_markdown,
    load_builtin_locked_model,
    refresh_models_for_custom,
    remember_custom_inputs,
    switch_model_source,
)
from .utils import decode_base64_to_image, encode_image_to_base64

_COPY_JS = """
(value) => {
    navigator.clipboard.writeText(value || "");
}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="ChatLLM Tester") as demo:
        gr.Markdown(
            "# ChatLLM Tester\n"
            "Giao diện chat để test model OpenAI-compatible, hỗ trợ nhiều ảnh upload hoặc paste."
        )

        with gr.Tabs():
            with gr.Tab("Chat", render_children=True):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=650, label="Chat")
                        message = gr.MultimodalTextbox(
                            label="Nội dung",
                            placeholder="Nhập câu hỏi, URL ảnh/video, YouTube URL, rồi upload hoặc paste ảnh/file/audio...",
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
                            choices=[],
                            value=None,
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
                        model_status = gr.Markdown(
                            value="Model có sẵn: đang tải danh sách model từ `/v1/models`..."
                        )
                        usage_status = gr.Markdown(value="**Token Usage:** _Chưa có dữ liệu_")
                        temperature = gr.Slider(
                            minimum=0, maximum=2, value=0.2, step=0.1, label="Temperature",
                        )
                        top_p = gr.Slider(
                            minimum=0, maximum=1, value=0.9, step=0.05, label="Top-P",
                        )
                        max_tokens = gr.Slider(
                            minimum=16, maximum=8192, value=1024, step=16, label="Max tokens",
                        )
                        system_prompt = gr.Textbox(
                            label="System prompt", value=DEFAULT_SYSTEM_PROMPT, lines=8,
                        )
                        gr.Markdown("**Gợi ý:** đổi Base URL và API key, sau đó tải model từ `/v1/models`.")

            with gr.Tab("Base64 Ảnh", render_children=True):
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
                copy_base64_btn.click(fn=None, inputs=[encoded_base64_output], js=_COPY_JS)
                copy_data_url_btn.click(fn=None, inputs=[encoded_data_url_output], js=_COPY_JS)
                decode_btn.click(
                    decode_base64_to_image,
                    inputs=[decode_base64_input],
                    outputs=[decoded_image_output, decoded_file_output, decode_status],
                )

            with gr.Tab("MarkItDown", render_children=True):
                gr.Markdown("Convert tài liệu (PDF, DOCX, XLSX, PPTX, HTML...) sang Markdown.")
                with gr.Row():
                    md_file_input = gr.File(label="Chọn file")
                    md_convert_btn = gr.Button("Convert sang Markdown", variant="primary")
                md_status = gr.Markdown(value="")
                md_output = gr.Textbox(label="Markdown", lines=25)

                md_convert_btn.click(
                    convert_to_markdown,
                    inputs=[md_file_input],
                    outputs=[md_output, md_status],
                )

        ui_state = gr.State([])
        api_state = gr.State([])
        session_id_state = gr.State(uuid.uuid4().hex[:12])
        custom_base_url_state = gr.State(env_any(["OPENAI_BASE_URL", "base_url"], "https://api.openai.com/v1"))
        custom_api_key_state = gr.State(env_any(["OPENAI_API_KEY", "api_key"], ""))
        custom_model_state = gr.State("")

        send_inputs = [
            message, ui_state, api_state, model_name, base_url, api_key,
            temperature, max_tokens, system_prompt, enable_thinking, top_p,
            session_id_state,
        ]
        send_outputs = [chatbot, ui_state, api_state, message, usage_status]

        send_btn.click(chat, inputs=send_inputs, outputs=send_outputs)
        message.submit(chat, inputs=send_inputs, outputs=send_outputs)
        demo.load(load_builtin_locked_model, inputs=[base_url, api_key], outputs=[model_name, model_status])

        # Remember custom inputs from any field change
        custom_inputs = [model_source, base_url, api_key, model_name, custom_base_url_state, custom_api_key_state, custom_model_state]
        custom_outputs = [custom_base_url_state, custom_api_key_state, custom_model_state]
        for component in (base_url, api_key, model_name):
            component.change(remember_custom_inputs, inputs=custom_inputs, outputs=custom_outputs)

        model_source.change(
            switch_model_source,
            inputs=[model_source, custom_base_url_state, custom_api_key_state, custom_model_state],
            outputs=[base_url, api_key, model_name, model_status],
        )

        refresh_models_btn.click(
            refresh_models_for_custom,
            inputs=[model_source, base_url, api_key, model_name, custom_base_url_state, custom_api_key_state, custom_model_state],
            outputs=[model_name, model_status, custom_base_url_state, custom_api_key_state, custom_model_state],
        )

        clear_btn.click(clear_chat, outputs=[chatbot, ui_state, api_state, message, usage_status])

    return demo
