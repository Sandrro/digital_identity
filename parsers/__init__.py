"""Парсеры текстов для разных источников."""

from .vk_group_parser import VKGroupParser
from .website_parser import parse_websites
from .yandex_reviews_parser import fetch_yandex_reviews

__all__ = [
    "fetch_yandex_reviews",
    "VKGroupParser",
    "parse_websites",
]
