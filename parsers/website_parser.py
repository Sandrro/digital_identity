"""Парсер контента со статических веб-страниц."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
PUNCT_RUN_RE = re.compile(r"([!?.,])\1{2,}")


def clean_text_minimal(text: str) -> str:
    if text is None:
        return ""
    value = str(text)
    value = HTML_TAG_RE.sub(" ", value)
    value = PUNCT_RUN_RE.sub(r"\1\1", value)
    value = SPACE_RE.sub(" ", value).strip()
    return value


def parse_date_safe(value: str) -> Optional[pd.Timestamp]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return pd.to_datetime(value)
    except Exception:
        return None


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        tag.decompose()
    for tag in soup.find_all(["nav", "footer", "header", "aside"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
        comment.extract()


def _get_title(soup: BeautifulSoup) -> Optional[str]:
    for attr, val in [("property", "og:title"), ("name", "twitter:title")]:
        node = soup.find("meta", attrs={attr: val})
        if node and node.get("content"):
            return clean_text_minimal(node["content"])

    if soup.title and soup.title.get_text(strip=True):
        return clean_text_minimal(soup.title.get_text(" ", strip=True))

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return clean_text_minimal(h1.get_text(" ", strip=True))

    return None


def _safe_to_datetime(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    dt = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(dt):
        return None
    return dt


def _get_date(soup: BeautifulSoup) -> Optional[pd.Timestamp]:
    meta_candidates = [
        ("property", "article:published_time"),
        ("property", "article:modified_time"),
        ("property", "og:updated_time"),
        ("name", "pubdate"),
        ("name", "publishdate"),
        ("name", "timestamp"),
        ("name", "date"),
        ("name", "DC.date.issued"),
        ("name", "DC.Date"),
        ("itemprop", "datePublished"),
        ("itemprop", "dateModified"),
    ]
    for attr, val in meta_candidates:
        node = soup.find("meta", attrs={attr: val})
        if node and node.get("content"):
            dt = _safe_to_datetime(node["content"])
            if dt is not None:
                return dt

    time_tag = soup.find("time")
    if time_tag:
        dt = _safe_to_datetime(time_tag.get("datetime"))
        if dt is not None:
            return dt
    return None


def _node_text_len(node: Any) -> int:
    if not hasattr(node, "get_text"):
        return 0
    return len(node.get_text(" ", strip=True))


def _extract_main_text(
    soup: BeautifulSoup, *, selector: Optional[str] = None, min_chars: int = 400
) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"extractor": None}

    if selector:
        nodes = soup.select(selector)
        if nodes:
            parts = [n.get_text("\n", strip=True) for n in nodes]
            text = clean_text_minimal("\n".join(parts))
            meta["extractor"] = f"css:{selector}"
            if len(text) >= min_chars:
                return text, meta
            meta["extractor_fallback"] = "too_short"

    for tag_name in ["article", "main"]:
        node = soup.find(tag_name)
        if node:
            text = clean_text_minimal(node.get_text("\n", strip=True))
            if len(text) >= min_chars:
                meta["extractor"] = tag_name
                return text, meta

    candidates: List[Any] = []
    for key in ["content", "article", "post", "entry", "text", "body", "main"]:
        candidates.extend(soup.find_all(attrs={"class": re.compile(key, re.I)}))
        candidates.extend(soup.find_all(attrs={"id": re.compile(key, re.I)}))

    candidates.extend(soup.find_all(["div", "section"]))

    best = None
    best_len = 0
    for node in candidates:
        length = _node_text_len(node)
        if length > best_len:
            best = node
            best_len = length

    if best is not None and best_len > 0:
        meta["extractor"] = "largest_block"
        text = clean_text_minimal(best.get_text("\n", strip=True))
        return text, meta

    meta["extractor"] = "none"
    return "", meta


@dataclass
class FetchConfig:
    timeout: int = 20
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome Safari"
    )
    max_bytes: int = 5_000_000


def _fetch_html(
    url: str, cfg: FetchConfig, session: requests.Session
) -> Tuple[Optional[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {"status_code": None, "final_url": None}
    try:
        response = session.get(
            url,
            headers={
                "User-Agent": cfg.user_agent,
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout=cfg.timeout,
            allow_redirects=True,
        )
        meta["status_code"] = response.status_code
        meta["final_url"] = response.url
        response.raise_for_status()

        content = response.content
        if content and len(content) > cfg.max_bytes:
            meta["error"] = f"response_too_large:{len(content)}"
            return None, meta

        if not response.encoding or response.encoding.lower() in {
            "iso-8859-1",
            "latin1",
            "ascii",
        }:
            if response.apparent_encoding:
                response.encoding = response.apparent_encoding

        return response.text, meta
    except Exception as exc:
        meta["error"] = repr(exc)
        return None, meta


def _stable_id(url: str, text: str) -> str:
    digest = hashlib.sha1(
        (url + "\n" + (text or "")[:4000]).encode("utf-8", errors="ignore")
    ).hexdigest()[:16]
    return f"web_{digest}"


def parse_websites(
    urls: List[str],
    *,
    selector: Optional[str] = None,
    min_chars: int = 400,
    cfg: Optional[FetchConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or FetchConfig()
    rows: List[Dict[str, Any]] = []

    with requests.Session() as session:
        for url in urls:
            url = (url or "").strip()
            if not url:
                continue

            html, fetch_meta = _fetch_html(url, cfg, session=session)
            if not html:
                rows.append(
                    {
                        "doc_id": _stable_id(url, ""),
                        "source": "website",
                        "text_raw": "",
                        "date": None,
                        "url": url,
                        "meta": {"fetch": fetch_meta},
                    }
                )
                continue

            soup = BeautifulSoup(html, "lxml")
            _strip_noise(soup)

            title = _get_title(soup)
            date = _get_date(soup)
            text_raw, extract_meta = _extract_main_text(
                soup, selector=selector, min_chars=min_chars
            )

            meta = {
                "fetch": fetch_meta,
                "title": title,
                "date_extracted": date.isoformat() if isinstance(date, pd.Timestamp) else None,
                "extraction": extract_meta,
                "domain": urlparse(fetch_meta.get("final_url") or url).netloc,
            }

            rows.append(
                {
                    "doc_id": _stable_id(fetch_meta.get("final_url") or url, text_raw),
                    "source": "website",
                    "text_raw": text_raw,
                    "date": date,
                    "url": fetch_meta.get("final_url") or url,
                    "meta": meta,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["doc_id", "source", "text_raw", "date", "url", "meta"])

    df["text_clean"] = df["text_raw"].map(clean_text_minimal)
    return df
