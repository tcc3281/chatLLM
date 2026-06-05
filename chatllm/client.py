from __future__ import annotations

from typing import Any

from openai import OpenAI

from .config import env
from .schemas import MediaUrl


def make_client(base_url: str, api_key: str, timeout: float | None = None) -> OpenAI:
    kwargs: dict[str, Any] = {}
    if timeout is not None:
        kwargs["timeout"] = timeout

    return OpenAI(
        base_url=base_url.strip() or "https://api.openai.com/v1",
        api_key=api_key.strip() or env("OPENAI_API_KEY", "EMPTY"),
        **kwargs,
    )


def resolve_model(model_name: str | None) -> str:
    model = (model_name or "").strip()
    if not model:
        raise ValueError("Chưa chọn model. Hãy tải danh sách model từ /v1/models trước.")
    return model


def list_models(base_url: str, api_key: str) -> list[str]:
    client = make_client(base_url, api_key, timeout=8.0)
    response = client.models.list()
    models = sorted({model.id for model in response.data if getattr(model, "id", None)})
    if not models:
        raise ValueError("/v1/models không trả về model nào.")
    return models


def _append_remote_media(content: list[dict[str, Any]], media_urls: list[MediaUrl]) -> None:
    for media in media_urls:
        media_type = media["type"]
        content.append(
            {
                "type": media_type,
                media_type: {"url": media["url"]},
            }
        )


def _append_uploaded_images(content: list[dict[str, Any]], image_paths: list[str]) -> None:
    from .utils import file_to_data_url

    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": file_to_data_url(image_path)},
            }
        )


def _log_outgoing_content(text: str, image_paths: list[str], media_urls: list[MediaUrl]) -> None:
    if text:
        print(f"[chatLLM] Sending text content (length: {len(text)})")
    if media_urls:
        image_count = sum(1 for media in media_urls if media["type"] == "image_url")
        video_count = sum(1 for media in media_urls if media["type"] == "video_url")
        print(f"[chatLLM] Sending remote media (images: {image_count}, videos: {video_count})")
    if image_paths:
        print(f"[chatLLM] Sending {len(image_paths)} uploaded images")


def build_user_content(
    text: str,
    image_paths: list[str],
    media_urls: list[MediaUrl],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    _append_remote_media(content, media_urls)
    _append_uploaded_images(content, image_paths)
    if text:
        content.append({"type": "text", "text": text})
    return content or [{"type": "text", "text": " "}]


def api_message_content(
    text: str,
    image_paths: list[str],
    media_urls: list[MediaUrl],
) -> str | list[dict[str, Any]]:
    _log_outgoing_content(text, image_paths, media_urls)

    if image_paths or media_urls:
        return build_user_content(text, image_paths, media_urls)
    return text or " "
