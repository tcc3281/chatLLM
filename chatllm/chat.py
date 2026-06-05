from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Generator

import gradio as gr
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai import Stream

from .client import api_message_content, make_client, resolve_model
from .config import DEFAULT_SYSTEM_PROMPT
from .utils import (
    extract_media_urls,
    get_youtube_info,
    merge_text_with_text_files,
    normalize_message,
    save_chat_history,
)


def _extract_youtube(text: str) -> str:
    """Replace YouTube URLs with their transcription embedded in XML tags."""
    yt_pattern = r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11})"
    for yt_url in re.findall(yt_pattern, text):
        try:
            print(f"[chatLLM] Extracting YouTube info from text: {yt_url}...")
            extracted = get_youtube_info(yt_url)
            if extracted:
                text = text.replace(yt_url, f"{yt_url}\n\n<youtube_transcription url=\"{yt_url}\">\n{extracted}\n</youtube_transcription>")
                print(f"[chatLLM] Successfully extracted YouTube info")
        except Exception as exc:
            print(f"[chatLLM] YouTube extraction error for {yt_url}: {exc}")
    return text


def _process_thinking_delta(
    content_delta: str,
    assistant_text: str,
    reasoning_text: str,
    in_manual_reasoning: bool,
) -> tuple[str, str, bool]:
    """Track thinking/reasoning content within the assistant's response."""
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

    return assistant_text, reasoning_text, in_manual_reasoning


def _build_usage_display(usage_info: Any) -> str:
    if not usage_info:
        return "**Token Usage:** _Chưa có dữ liệu_"
    return (
        f"**Token Usage:** Prompt: {usage_info.prompt_tokens} | "
        f"Completion: {usage_info.completion_tokens} | "
        f"Total: {usage_info.total_tokens}"
    )


def build_display_content(
    text: str,
    image_paths: list[str],
    other_file_paths: list[str],
    media_urls: list[dict[str, str]],
) -> str | list[Any]:
    if not image_paths and not other_file_paths and not media_urls:
        return text or "[Tin nhắn trống]"

    content: list[Any] = []

    if other_file_paths:
        file_pills = "".join(f"📎 **{Path(fp).name}** &nbsp; " for fp in other_file_paths)
        content.append(file_pills.strip())

    if text:
        content.append(text)

    if media_urls:
        media_lines = []
        for media in media_urls:
            label = "Image URL" if media["type"] == "image_url" else "Video URL"
            media_lines.append(f"- **{label}:** {media['url']}")
        content.append("\n".join(media_lines))

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
) -> Generator[tuple[list, list, list, dict, str], None, None]:
    def _clear_input() -> dict[str, Any]:
        return {"text": "", "files": []}

    text, file_paths = normalize_message(message)
    text = _extract_youtube(text)
    text, media_urls = extract_media_urls(text)
    text, image_paths, other_file_paths = merge_text_with_text_files(text, file_paths)

    if not text and not image_paths and not media_urls:
        raise gr.Error("Nhập nội dung, URL ảnh/video hoặc tải ảnh lên trước khi gửi.")

    for image_path in image_paths:
        if not Path(image_path).exists():
            raise gr.Error(f"Không tìm thấy file ảnh: {image_path}")

    try:
        model = resolve_model(model_name)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    client = make_client(base_url, api_key)
    display_content = build_display_content(text, image_paths, other_file_paths, media_urls)

    ui_history = list(ui_history) + [
        {"role": "user", "content": display_content},
        {"role": "assistant", "content": ""},
    ]

    user_content = api_message_content(text, image_paths, media_urls)
    request_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT},
        *api_history,
        {"role": "user", "content": user_content},
    ]

    extra_body: dict[str, Any] = {}
    if enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}

    stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
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
            assistant_text, reasoning_text, in_manual_reasoning = _process_thinking_delta(
                content_delta, assistant_text, reasoning_text, in_manual_reasoning
            )

        display_text = assistant_text
        if reasoning_text:
            display_text = f"<details open>\n<summary>Thinking Process</summary>\n\n{reasoning_text}\n</details>\n\n{assistant_text}"

        ui_history[-1]["content"] = display_text or "..."

        yield ui_history, ui_history, api_history, _clear_input(), _build_usage_display(usage_info)

    assistant_text = assistant_text.strip() or (
        reasoning_text.strip() and "(thinking completed, no content)"
    ) or "(không có nội dung trả về)"

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
            }
            if usage_info
            else None,
        },
        messages=api_history[-2:],
    )

    yield ui_history, ui_history, api_history, _clear_input(), _build_usage_display(usage_info)
