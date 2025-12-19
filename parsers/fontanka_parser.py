"""Парсер материалов с Fontanka.ru."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class FontankaArticle:
    url: str
    title: str
    published_at: Optional[str]
    text: str


def fetch_fontanka_article(url: str, timeout: int = 20) -> FontankaArticle:
    """Загружает и парсит статью Fontanka.ru по URL.

    Args:
        url: ссылка на материал.
        timeout: таймаут запроса.

    Returns:
        FontankaArticle: объект с заголовком, датой и текстом.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    title_tag = soup.select_one("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    time_tag = soup.select_one("time")
    published_at = time_tag.get("datetime") if time_tag else None

    body = (
        soup.select_one('[itemprop="articleBody"]')
        or soup.select_one(".article__text")
        or soup.select_one(".article__body")
    )
    if body:
        paragraphs = [p.get_text(" ", strip=True) for p in body.find_all("p")]
        text = "\n".join([p for p in paragraphs if p])
    else:
        text = ""

    return FontankaArticle(url=url, title=title, published_at=published_at, text=text)
