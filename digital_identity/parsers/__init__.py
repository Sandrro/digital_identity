"""Совместимость для импорта парсеров через пакет digital_identity."""

from importlib import import_module

_parsers = import_module("parsers")

fetch_fontanka_article = _parsers.fetch_fontanka_article
fetch_yandex_reviews = _parsers.fetch_yandex_reviews
VKGroupParser = _parsers.VKGroupParser

__all__ = [
    "fetch_fontanka_article",
    "fetch_yandex_reviews",
    "VKGroupParser",
]
