from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from wordcloud import STOPWORDS, WordCloud

RUSSIAN_STOPWORDS = {
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "на",
    "я",
    "с",
    "со",
    "как",
    "а",
    "то",
    "все",
    "она",
    "так",
    "его",
    "но",
    "да",
    "ты",
    "к",
    "у",
    "же",
    "вы",
    "за",
    "бы",
    "по",
    "только",
    "ее",
    "мне",
    "было",
    "вот",
    "от",
    "меня",
    "еще",
    "нет",
    "о",
    "из",
    "ему",
    "теперь",
    "когда",
    "даже",
    "ну",
    "вдруг",
    "ли",
    "если",
    "уже",
    "или",
    "ни",
    "быть",
    "был",
    "него",
    "до",
    "вас",
    "нибудь",
    "опять",
    "уж",
    "вам",
    "ведь",
    "там",
    "потом",
    "себя",
    "ничего",
    "ей",
    "может",
    "они",
    "тут",
    "где",
    "есть",
    "надо",
    "ней",
    "для",
    "мы",
    "тебя",
    "их",
    "чем",
    "была",
    "сам",
    "чтоб",
    "без",
    "будто",
    "чего",
    "раз",
    "тоже",
    "себе",
    "под",
    "будет",
    "ж",
    "тогда",
    "кто",
    "этот",
    "того",
    "потому",
    "этого",
    "какой",
    "совсем",
    "ним",
    "здесь",
    "этом",
    "один",
    "почти",
    "мой",
    "тем",
    "чтобы",
    "нее",
    "сейчас",
    "были",
    "куда",
    "зачем",
    "всех",
    "никогда",
    "можно",
    "при",
    "наконец",
    "два",
    "об",
    "другой",
    "хоть",
    "после",
    "над",
    "больше",
    "тот",
    "через",
    "эти",
    "нас",
    "про",
    "всего",
    "них",
    "какая",
    "много",
    "разве",
    "три",
    "эту",
    "моя",
    "впрочем",
    "хорошо",
    "свою",
    "этой",
    "перед",
    "иногда",
    "лучше",
    "чуть",
    "том",
    "нельзя",
    "такой",
    "им",
    "более",
    "всегда",
    "конечно",
    "всю",
    "между",
}

TOKEN_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё]+", re.UNICODE)


@dataclass
class WordCloudResult:
    wordcloud: WordCloud
    cleaned_text: str


def parse_stop_words(raw_value: str | Sequence[str] | None) -> set[str]:
    if raw_value is None:
        return set()
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return set()
        tokens = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
        stop_words: set[str] = set()
        for token in tokens:
            if token == "english":
                stop_words.update(STOPWORDS)
            elif token == "russian":
                stop_words.update(RUSSIAN_STOPWORDS)
            else:
                stop_words.add(token)
        return stop_words
    stop_words = set()
    for item in raw_value:
        token = str(item).strip().lower()
        if not token:
            continue
        if token == "english":
            stop_words.update(STOPWORDS)
        elif token == "russian":
            stop_words.update(RUSSIAN_STOPWORDS)
        else:
            stop_words.add(token)
    return stop_words


def _clean_texts(texts: Iterable[str], stop_words: set[str]) -> str:
    tokens: list[str] = []
    for text in texts:
        if not text:
            continue
        for token in TOKEN_PATTERN.findall(str(text).lower()):
            if token in stop_words:
                continue
            tokens.append(token)
    return " ".join(tokens)


def build_wordcloud(
    texts: Iterable[str],
    *,
    stop_words: str | Sequence[str] | None = None,
    width: int = 800,
    height: int = 400,
    background_color: str = "white",
    max_words: int = 200,
    collocations: bool = False,
    **kwargs,
) -> WordCloudResult:
    stop_words_set = parse_stop_words(stop_words)
    cleaned_text = _clean_texts(texts, stop_words_set)
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        collocations=collocations,
        stopwords=stop_words_set,
        **kwargs,
    ).generate(cleaned_text)
    return WordCloudResult(wordcloud=wordcloud, cleaned_text=cleaned_text)
