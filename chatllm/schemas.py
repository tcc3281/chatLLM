from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

MediaType = Literal["image_url", "video_url"]


class MediaUrl(TypedDict):
    type: MediaType
    url: str


@dataclass(frozen=True)
class PreparedUserMessage:
    text: str
    image_paths: list[str]
    other_file_paths: list[str]
    media_urls: list[MediaUrl]


@dataclass(frozen=True)
class ChatSettings:
    model: str
    base_url: str
    temperature: float
    top_p: float
    max_tokens: int
    system_prompt: str
    enable_thinking: bool


@dataclass
class StreamState:
    assistant_text: str = ""
    reasoning_text: str = ""
    in_manual_reasoning: bool = False
    usage_info: Any = None
