from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse
from datetime import datetime

import requests


@dataclass
class VKPost:
    post_id: int
    date: int
    text: str
    attachments: List[Dict]


@dataclass
class VKComment:
    comment_id: int
    post_id: int
    from_id: int
    date: int
    text: str
    likes_count: int
    parent_id: Optional[int]


def _normalize_group_id(group: str) -> str:
    raw = group.strip()
    if raw.startswith(("https://", "http://")):
        parsed = urlparse(raw)
        path = parsed.path.rstrip("/")
        raw = path.split("/")[-1] if path else ""
    if "vk.com/" in raw:
        raw = raw.split("vk.com/")[-1]
    raw = raw.lstrip("@")
    raw = raw.split("?")[0].split("#")[0]
    return raw


class VKGroupParser:
    """Парсер постов и комментариев сообществ ВК."""

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

    # ---------- POSTS ----------

    def fetch_wall_posts(
        self,
        group: str,
        count: int = 100,
        offset: int = 0,
        owner_id: Optional[int] = None,
    ) -> List[VKPost]:

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
                    post_id=item["id"],
                    date=item["date"],
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

        fetched = 0
        offset = 0
        while fetched < total:
            batch = self.fetch_wall_posts(
                group,
                count=min(100, total - fetched),
                offset=offset,
            )
            if not batch:
                break
            for post in batch:
                yield post
            fetched += len(batch)
            offset += len(batch)
            time.sleep(sleep_seconds)

    # ---------- COMMENTS ----------

    def _fetch_comments_batch(
        self,
        owner_id: int,
        post_id: int,
        offset: int = 0,
        comment_id: Optional[int] = None,
    ) -> List[VKComment]:

        params = {
            "owner_id": owner_id,
            "post_id": post_id,
            "count": 100,
            "offset": offset,
            "need_likes": 1,
            "extended": 0,
        }

        if comment_id is not None:
            params["comment_id"] = comment_id

        response = self._request("wall.getComments", params)

        comments: List[VKComment] = []

        for item in response.get("items", []):
            comments.append(
                VKComment(
                    comment_id=item["id"],
                    post_id=post_id,
                    from_id=item["from_id"],
                    date=item["date"],
                    text=item.get("text", ""),
                    likes_count=item.get("likes", {}).get("count", 0),
                    parent_id=comment_id,
                )
            )

            # вложенные комментарии
            thread = item.get("thread", {})
            if thread.get("count", 0) > 0:
                subcomments = self._fetch_comments_batch(
                    owner_id=owner_id,
                    post_id=post_id,
                    comment_id=item["id"],
                )
                comments.extend(subcomments)

        return comments

    def fetch_post_comments(
        self,
        owner_id: int,
        post_id: int,
        sleep_seconds: float = 0.35,
    ) -> List[VKComment]:

        all_comments: List[VKComment] = []
        offset = 0

        while True:
            batch = self._fetch_comments_batch(
                owner_id=owner_id,
                post_id=post_id,
                offset=offset,
            )
            if not batch:
                break
            all_comments.extend(batch)
            offset += len(batch)
            time.sleep(sleep_seconds)

        return all_comments

    def iter_post_comments(
        self,
        owner_id: int,
        post_id: int,
    ) -> Iterable[VKComment]:

        comments = self.fetch_post_comments(owner_id, post_id)
        for comment in comments:
            yield comment

    # ---------- POSTS + COMMENTS ----------

    def iter_posts_with_comments(
        self,
        group: str,
        total_posts: int = 100,
    ) -> Iterable[tuple[VKPost, List[VKComment]]]:

        domain = _normalize_group_id(group)
        info = self._request("groups.getById", {"group_id": domain})
        owner_id = -int(info[0]["id"])

        for post in self.iter_posts(group, total=total_posts):
            comments = self.fetch_post_comments(owner_id, post.post_id)
            yield post, comments
