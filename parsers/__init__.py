"""Парсеры текстов для разных источников."""

from .fontanka_parser import fetch_fontanka_article
from .vk_group_parser import VKGroupParser
from .yandex_reviews_parser import fetch_yandex_reviews

__all__ = [
    "fetch_fontanka_article",
    "fetch_yandex_reviews",
    "VKGroupParser",
]
