from __future__ import annotations

from typing import Any

import gradio as gr
from openai import OpenAI

from .config import DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, env
from .utils import (
    file_to_data_url,
    save_chat_history,
)


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=base_url.strip() or "https://api.openai.com/v1",
        api_key=api_key.strip() or env("OPENAI_API_KEY", "EMPTY"),
    )


def resolve_model(model_name: str) -> str:
    return model_name.strip() or DEFAULT_MODEL


def list_models(base_url: str, api_key: str) -> list[str]:
    client = make_client(base_url, api_key)
    response = client.models.list()
    models = sorted({model.id for model in response.data if getattr(model, "id", None)})
    return models or [DEFAULT_MODEL]


def load_builtin_locked_model(base_url: str, api_key: str):
    try:
        models = list_models(base_url, api_key)
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
        models = list_models(base_url, api_key)
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


def build_user_content(text: str, image_paths: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": file_to_data_url(image_path)},
            }
        )
    return content or [{"type": "text", "text": " "}]


def api_message_content(text: str, image_paths: list[str]) -> str | list[dict[str, Any]]:
    if text:
        print(f"[chatLLM] Sending text content (length: {len(text)})")
    if image_paths:
        print(f"[chatLLM] Sending {len(image_paths)} images")

    if image_paths:
        return build_user_content(text, image_paths)
    return text or " "


def build_display_content(text: str, image_paths: list[str], other_file_paths: list[str]) -> str | list[Any]:
    from pathlib import Path
    if not image_paths and not other_file_paths:
        return text or "[Tin nhắn trống]"

    content: list[Any] = []
    
    if other_file_paths:
        file_pills = ""
        for fp in other_file_paths:
            name = Path(fp).name
            file_pills += f"📎 **{name}** &nbsp; "
        content.append(file_pills.strip())

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
    enable_thinking: bool = False,
    top_p: float = 0.9,
    session_id: str = "",
):
    from .utils import normalize_message, merge_text_with_text_files
    from pathlib import Path

    def _clear_input() -> dict[str, Any]:
        return {"text": "", "files": []}

    text, file_paths = normalize_message(message)
    
    # Extract and process YouTube URLs from text
    import re
    yt_pattern = r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11})"
    yt_matches = re.findall(yt_pattern, text)
    for yt_url in yt_matches:
        try:
            from .utils import get_youtube_info
            print(f"[chatLLM] Extracting YouTube info from text: {yt_url}...")
            extracted = get_youtube_info(yt_url)
            if extracted:
                text = text.replace(yt_url, f"{yt_url}\n\n<youtube_transcription url=\"{yt_url}\">\n{extracted}\n</youtube_transcription>")
                print(f"[chatLLM] Successfully extracted YouTube info")
        except Exception as exc:
            print(f"[chatLLM] YouTube extraction error for {yt_url}: {exc}")

    text, image_paths, other_file_paths = merge_text_with_text_files(text, file_paths)
    
    if not text and not image_paths:
        raise gr.Error("Nhập nội dung hoặc tải ảnh lên trước khi gửi.")

    for image_path in image_paths:
        if not Path(image_path).exists():
            raise gr.Error(f"Không tìm thấy file ảnh: {image_path}")

    model = resolve_model(model_name)
    client = make_client(base_url, api_key)

    display_content = build_display_content(text, image_paths, other_file_paths)

    ui_history = list(ui_history) + [
        {"role": "user", "content": display_content},
        {"role": "assistant", "content": ""}
    ]

    user_content = api_message_content(text, image_paths)
    request_messages = [
        {"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT},
        *api_history,
        {"role": "user", "content": user_content},
    ]

    extra_body = {}
    if enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}

    stream = client.chat.completions.create(
        model=model,
        messages=request_messages,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra_body,
    )

    assistant_text = ""
    reasoning_text = ""
    in_manual_reasoning = False
    usage_info = None

    for chunk in stream:
        if hasattr(chunk, "usage") and chunk.usage:
            usage_info = chunk.usage
            continue

        if not chunk.choices:
            continue
            
        delta = chunk.choices[0].delta
        
        reasoning_delta = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        if reasoning_delta:
            reasoning_text += reasoning_delta
        
        content_delta = delta.content or ""
        if content_delta:
            temp_content = assistant_text + content_delta
            
            if not in_manual_reasoning and "Thinking Process:" in temp_content:
                in_manual_reasoning = True
                parts = temp_content.split("Thinking Process:", 1)
                assistant_text = parts[0]
                reasoning_text += "Thinking Process:" + parts[1]
            elif in_manual_reasoning:
                reasoning_text += content_delta
            else:
                assistant_text += content_delta

        display_text = assistant_text
        if reasoning_text:
            display_text = f"<details open>\n<summary>Thinking Process</summary>\n\n{reasoning_text}\n</details>\n\n{assistant_text}"

        ui_history[-1]["content"] = display_text or "..."
        
        usage_display = ""
        if usage_info:
            usage_display = (
                f"**Token Usage:** Prompt: {usage_info.prompt_tokens} | "
                f"Completion: {usage_info.completion_tokens} | "
                f"Total: {usage_info.total_tokens}"
            )
        
        yield ui_history, ui_history, api_history, _clear_input(), usage_display

    assistant_text = assistant_text.strip() or (reasoning_text.strip() and "(thinking completed, no content)") or "(không có nội dung trả về)"
    
    final_display = assistant_text
    if reasoning_text:
        final_display = f"<details>\n<summary>Thinking Process (Completed)</summary>\n\n{reasoning_text}\n</details>\n\n{assistant_text}"

    ui_history[-1]["content"] = final_display
    api_history = list(api_history) + [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_text},
    ]

    save_chat_history(
        session_id=session_id,
        config={
            "model": model,
            "base_url": base_url,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "enable_thinking": enable_thinking,
            "usage": {
                "prompt_tokens": usage_info.prompt_tokens if usage_info else 0,
                "completion_tokens": usage_info.completion_tokens if usage_info else 0,
                "total_tokens": usage_info.total_tokens if usage_info else 0,
            } if usage_info else None
        },
        messages=api_history[-2:],
    )

    usage_display = ""
    if usage_info:
        usage_display = (
            f"**Token Usage:** Prompt: {usage_info.prompt_tokens} | "
            f"Completion: {usage_info.completion_tokens} | "
            f"Total: {usage_info.total_tokens}"
        )

    yield ui_history, ui_history, api_history, _clear_input(), usage_display
