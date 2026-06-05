from __future__ import annotations

from typing import Any

from openai import OpenAI

from .config import env


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


def build_user_content(
    text: str,
    image_paths: list[str],
    media_urls: list[dict[str, str]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for media in media_urls:
        media_type = media["type"]
        content.append(
            {
                "type": media_type,
                media_type: {"url": media["url"]},
            }
        )

    for image_path in image_paths:
        from .utils import file_to_data_url
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": file_to_data_url(image_path)},
            }
        )
    if text:
        content.append({"type": "text", "text": text})
    return content or [{"type": "text", "text": " "}]


def api_message_content(
    text: str,
    image_paths: list[str],
    media_urls: list[dict[str, str]],
) -> str | list[dict[str, Any]]:
    if text:
        print(f"[chatLLM] Sending text content (length: {len(text)})")
    if media_urls:
        counts = {"image_url": 0, "video_url": 0}
        for media in media_urls:
            counts[media["type"]] += 1
        print(
            "[chatLLM] Sending remote media "
            f"(images: {counts['image_url']}, videos: {counts['video_url']})"
        )
    if image_paths:
        print(f"[chatLLM] Sending {len(image_paths)} uploaded images")

    if image_paths or media_urls:
        return build_user_content(text, image_paths, media_urls)
    return text or " "
