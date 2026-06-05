from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Generator

import gradio as gr
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai import Stream

from .client import api_message_content, make_client, resolve_model
from .config import DEFAULT_SYSTEM_PROMPT
from .schemas import ChatSettings, MediaUrl, PreparedUserMessage, StreamState
from .utils import (
    extract_media_urls,
    get_youtube_info,
    merge_text_with_text_files,
    normalize_message,
    save_chat_history,
)


def _clear_input() -> dict[str, Any]:
    return {"text": "", "files": []}


def _youtube_transcription_block(url: str, content: str) -> str:
    return f"<youtube_transcription url=\"{url}\">\n{content}\n</youtube_transcription>"


def _extract_youtube(text: str) -> str:
    """Replace YouTube URLs with their transcription embedded in XML tags."""
    yt_pattern = r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[a-zA-Z0-9_-]{11})"
    for yt_url in re.findall(yt_pattern, text):
        try:
            print(f"[chatLLM] Extracting YouTube info from text: {yt_url}...")
            extracted = get_youtube_info(yt_url)
            if extracted:
                text = text.replace(
                    yt_url,
                    f"{yt_url}\n\n{_youtube_transcription_block(yt_url, extracted)}",
                )
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
    media_urls: list[MediaUrl],
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


def _prepare_user_message(message: Any) -> PreparedUserMessage:
    text, file_paths = normalize_message(message)
    text = _extract_youtube(text)
    text, media_urls = extract_media_urls(text)
    text, image_paths, other_file_paths = merge_text_with_text_files(text, file_paths)
    return PreparedUserMessage(
        text=text,
        image_paths=image_paths,
        other_file_paths=other_file_paths,
        media_urls=media_urls,
    )


def _validate_user_message(user_message: PreparedUserMessage) -> None:
    if not user_message.text and not user_message.image_paths and not user_message.media_urls:
        raise gr.Error("Nhập nội dung, URL ảnh/video hoặc tải ảnh lên trước khi gửi.")

    for image_path in user_message.image_paths:
        if not Path(image_path).exists():
            raise gr.Error(f"Không tìm thấy file ảnh: {image_path}")


def _resolve_model_for_ui(model_name: str) -> str:
    try:
        return resolve_model(model_name)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def _build_request_messages(
    system_prompt: str,
    api_history: list[dict[str, Any]],
    user_content: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT},
        *api_history,
        {"role": "user", "content": user_content},
    ]


def _build_extra_body(enable_thinking: bool) -> dict[str, Any]:
    if not enable_thinking:
        return {}
    return {"chat_template_kwargs": {"enable_thinking": True}}


def _format_reasoning_display(
    assistant_text: str,
    reasoning_text: str,
    *,
    is_complete: bool,
) -> str:
    if not reasoning_text:
        return assistant_text

    if is_complete:
        summary = "Thinking Process (Completed)"
        open_attr = ""
    else:
        summary = "Thinking Process"
        open_attr = " open"

    return (
        f"<details{open_attr}>\n"
        f"<summary>{summary}</summary>\n\n"
        f"{reasoning_text}\n"
        f"</details>\n\n{assistant_text}"
    )


def _update_stream_state(state: StreamState, chunk: ChatCompletionChunk) -> bool:
    if hasattr(chunk, "usage") and chunk.usage:
        state.usage_info = chunk.usage
        return False

    if not chunk.choices:
        return False

    delta = chunk.choices[0].delta

    reasoning_delta = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
    if reasoning_delta:
        state.reasoning_text += reasoning_delta

    content_delta = delta.content or ""
    if content_delta:
        state.assistant_text, state.reasoning_text, state.in_manual_reasoning = _process_thinking_delta(
            content_delta,
            state.assistant_text,
            state.reasoning_text,
            state.in_manual_reasoning,
        )
    return True


def _final_assistant_text(state: StreamState) -> str:
    return state.assistant_text.strip() or (
        state.reasoning_text.strip() and "(thinking completed, no content)"
    ) or "(không có nội dung trả về)"


def _usage_payload(usage_info: Any) -> dict[str, int] | None:
    if not usage_info:
        return None
    return {
        "prompt_tokens": usage_info.prompt_tokens,
        "completion_tokens": usage_info.completion_tokens,
        "total_tokens": usage_info.total_tokens,
    }


def _save_chat_turn(
    session_id: str,
    settings: ChatSettings,
    usage_info: Any,
    messages: list[dict[str, Any]],
) -> None:
    save_chat_history(
        session_id=session_id,
        config={
            "model": settings.model,
            "base_url": settings.base_url,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens,
            "system_prompt": settings.system_prompt,
            "enable_thinking": settings.enable_thinking,
            "usage": _usage_payload(usage_info),
        },
        messages=messages,
    )


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
    user_message = _prepare_user_message(message)
    _validate_user_message(user_message)

    settings = ChatSettings(
        model=_resolve_model_for_ui(model_name),
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )
    client = make_client(settings.base_url, api_key)
    display_content = build_display_content(
        user_message.text,
        user_message.image_paths,
        user_message.other_file_paths,
        user_message.media_urls,
    )

    ui_history = list(ui_history) + [
        {"role": "user", "content": display_content},
        {"role": "assistant", "content": ""},
    ]

    user_content = api_message_content(
        user_message.text,
        user_message.image_paths,
        user_message.media_urls,
    )

    stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
        model=settings.model,
        messages=_build_request_messages(settings.system_prompt, api_history, user_content),
        temperature=float(settings.temperature),
        top_p=float(settings.top_p),
        max_tokens=int(settings.max_tokens),
        stream=True,
        stream_options={"include_usage": True},
        extra_body=_build_extra_body(settings.enable_thinking),
    )

    stream_state = StreamState()

    for chunk in stream:
        if not _update_stream_state(stream_state, chunk):
            continue
        display_text = _format_reasoning_display(
            stream_state.assistant_text,
            stream_state.reasoning_text,
            is_complete=False,
        )
        ui_history[-1]["content"] = display_text or "..."
        yield (
            ui_history,
            ui_history,
            api_history,
            _clear_input(),
            _build_usage_display(stream_state.usage_info),
        )

    assistant_text = _final_assistant_text(stream_state)
    final_display = _format_reasoning_display(
        assistant_text,
        stream_state.reasoning_text,
        is_complete=True,
    )

    ui_history[-1]["content"] = final_display
    api_history = list(api_history) + [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_text},
    ]

    _save_chat_turn(
        session_id=session_id,
        settings=settings,
        usage_info=stream_state.usage_info,
        messages=api_history[-2:],
    )

    yield (
        ui_history,
        ui_history,
        api_history,
        _clear_input(),
        _build_usage_display(stream_state.usage_info),
    )
