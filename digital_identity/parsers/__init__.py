"""Совместимость для импорта парсеров через пакет digital_identity."""

from importlib import import_module

_parsers = import_module("parsers")

fetch_yandex_reviews = _parsers.fetch_yandex_reviews
parse_websites = _parsers.parse_websites
VKGroupParser = _parsers.VKGroupParser

__all__ = [
    "fetch_yandex_reviews",
    "VKGroupParser",
    "parse_websites",
]
