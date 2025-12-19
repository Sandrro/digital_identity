"""Парсеры текстов для разных источников."""

from .vk_group_parser import VKGroupParser
from .website_parser import parse_websites

__all__ = [
    "VKGroupParser",
    "parse_websites",
]
