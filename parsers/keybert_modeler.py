from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

from keybert import KeyBERT
from tqdm.auto import tqdm

KeywordResult = List[Tuple[str, float]]


@dataclass
class KeyBERTResult:
    model: KeyBERT
    keywords: List[KeywordResult]


def extract_keywords(
    texts: Iterable[str],
    *,
    embedding_model: str | None = None,
    top_n: int = 5,
    keyphrase_ngram_range: tuple[int, int] = (1, 2),
    stop_words: str | Sequence[str] | None = None,
    use_mmr: bool = False,
    diversity: float = 0.5,
    use_maxsum: bool = False,
    nr_candidates: int = 20,
    show_progress: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> KeyBERTResult:
    model = KeyBERT(embedding_model)
    keywords = []
    texts_list = list(texts)
    iterator = (
        tqdm(texts_list, total=len(texts_list), desc="KeyBERT")
        if show_progress
        else texts_list
    )
    for idx, text in enumerate(iterator, start=1):
        if not text:
            keywords.append([])
            continue
        extracted = model.extract_keywords(
            text,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words=stop_words,
            top_n=top_n,
            use_mmr=use_mmr,
            diversity=diversity,
            use_maxsum=use_maxsum,
            nr_candidates=nr_candidates,
        )
        keywords.append(list(extracted))
        if progress_callback:
            progress_callback(idx, len(texts_list))
    return KeyBERTResult(model=model, keywords=keywords)


def format_keywords(keywords: KeywordResult, *, joiner: str = ", ") -> str:
    return joiner.join(keyword for keyword, _ in keywords)


def attach_keywords(
    data,
    keywords: Iterable[KeywordResult],
    *,
    keywords_column: str = "keywords",
    scores_column: str = "keyword_scores",
    joiner: str = ", ",
):
    result = data.copy()
    keywords_list = list(keywords)
    result[keywords_column] = [
        format_keywords(keyword_list, joiner=joiner)
        for keyword_list in keywords_list
    ]
    result[scores_column] = [
        [score for _, score in keyword_list] for keyword_list in keywords_list
    ]
    return result
