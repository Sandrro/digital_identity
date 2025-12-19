"""Парсер отзывов с Яндекс.Карт/Яндекс.Отзывов.

Из-за динамической загрузки отзывы могут быть недоступны без API/браузера.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class YandexReview:
    author: Optional[str]
    rating: Optional[float]
    text: str


def _extract_review_blocks(raw_html: str) -> Iterable[dict]:
    """Пробует достать JSON-блоки с отзывами из HTML."""
    review_blocks = []

    for script in re.findall(r"window\.__INITIAL_STATE__\s*=\s*({.*?})\s*;", raw_html, re.S):
        try:
            data = json.loads(script)
        except json.JSONDecodeError:
            continue

        review_blocks.append(data)

    return review_blocks


def _extract_reviews_from_state(state: dict) -> List[YandexReview]:
    reviews: List[YandexReview] = []
    text_candidates = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"text", "reviewText", "comment"} and isinstance(value, str):
                    text_candidates.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(state)

    for text in text_candidates:
        cleaned = html.unescape(text)
        if cleaned and cleaned not in {r.text for r in reviews}:
            reviews.append(YandexReview(author=None, rating=None, text=cleaned))

    return reviews


def fetch_yandex_reviews(url: str, timeout: int = 20, max_reviews: int = 20) -> List[YandexReview]:
    """Загружает отзывы с Яндекс.Карт/Яндекс.Отзывов по URL.

    Args:
        url: ссылка на страницу организации/отзывов.
        timeout: таймаут запроса.
        max_reviews: максимальное число отзывов в выдаче.

    Returns:
        Список YandexReview (текст может быть пустым при ограничениях доступа).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    html_text = response.text
    soup = BeautifulSoup(html_text, "lxml")

    reviews = []
    for block in _extract_review_blocks(html_text):
        reviews.extend(_extract_reviews_from_state(block))

    if reviews:
        return reviews[:max_reviews]

    text_nodes = soup.select("span[class*='review-text'], div[class*='review-text']")
    for node in text_nodes:
        text = node.get_text(" ", strip=True)
        if text:
            reviews.append(YandexReview(author=None, rating=None, text=text))
            if len(reviews) >= max_reviews:
                break

    return reviews
