"""Парсер постов групп ВК через API.

Адаптировано под актуальный Python в Google Colab.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import requests


@dataclass
class VKPost:
    post_id: int
    date: int
    text: str
    attachments: List[Dict]


def _normalize_group_id(group: str) -> str:
    if group.startswith("https://"):
        return group.rstrip("/").split("/")[-1]
    return group.lstrip("@")


class VKGroupParser:
    """Парсер постов из стены сообщества ВК."""

    def __init__(self, token: str, api_version: str = "5.131") -> None:
        self.token = token
        self.api_version = api_version

    def _request(self, method: str, params: Dict) -> Dict:
        url = f"https://api.vk.com/method/{method}"
        payload = {
            **params,
            "access_token": self.token,
            "v": self.api_version,
        }
        response = requests.get(url, params=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"VK API error: {data['error']}")
        return data["response"]

    def fetch_wall_posts(
        self,
        group: str,
        count: int = 100,
        offset: int = 0,
        owner_id: Optional[int] = None,
    ) -> List[VKPost]:
        """Возвращает список постов сообщества.

        Args:
            group: домен или ссылка на группу.
            count: количество постов за запрос (до 100).
            offset: смещение.
            owner_id: числовой id сообщества (если известен).
        """
        if owner_id is None:
            domain = _normalize_group_id(group)
            info = self._request("groups.getById", {"group_id": domain})
            owner_id = -int(info[0]["id"])

        response = self._request(
            "wall.get",
            {
                "owner_id": owner_id,
                "count": min(count, 100),
                "offset": offset,
                "filter": "owner",
            },
        )

        posts: List[VKPost] = []
        for item in response.get("items", []):
            posts.append(
                VKPost(
                    post_id=item.get("id"),
                    date=item.get("date"),
                    text=item.get("text", ""),
                    attachments=item.get("attachments", []),
                )
            )
        return posts

    def iter_posts(
        self,
        group: str,
        total: int = 200,
        sleep_seconds: float = 0.35,
    ) -> Iterable[VKPost]:
        """Итерирует посты сообщества.

        Args:
            group: домен или ссылка на группу.
            total: сколько постов получить.
            sleep_seconds: задержка между запросами.
        """
        fetched = 0
        offset = 0
        while fetched < total:
            batch = self.fetch_wall_posts(group, count=min(100, total - fetched), offset=offset)
            if not batch:
                break
            for post in batch:
                yield post
            fetched += len(batch)
            offset += len(batch)
            time.sleep(sleep_seconds)
