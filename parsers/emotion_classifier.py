import logging
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from transformers import pipeline
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

MAX_MODEL_TOKENS = 512
TIE_BREAK_PRIORITY = ["negative", "positive", "neutral"]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def _split_by_token_slices(tokenizer, text: str, max_tokens: int) -> List[str]:

    text = text.strip()
    if not text:
        return []

    if _token_len(tokenizer, text) <= max_tokens:
        return [text]

    slice_size = max(1, max_tokens - 2)
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: List[str] = []
    for i in range(0, len(ids), slice_size):
        sub = tokenizer.decode(ids[i : i + slice_size], skip_special_tokens=True).strip()
        if sub:
            chunks.append(sub)

    safe_chunks = []
    for c in chunks:
        if _token_len(tokenizer, c) <= max_tokens:
            safe_chunks.append(c)
        else:
            safe_chunks.extend(_split_by_token_slices(tokenizer, c, max_tokens))
    return safe_chunks


def _chunk_text_by_sentences(tokenizer, text: str, max_tokens: int = MAX_MODEL_TOKENS) -> List[str]:
    text = _normalize_ws(text)
    if not text:
        return []

    sentences = _split_into_sentences(text)
    if not sentences:
        return _split_by_token_slices(tokenizer, text, max_tokens)

    chunks: List[str] = []
    current: List[str] = []

    def flush_current() -> None:
        nonlocal current
        if current:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if _token_len(tokenizer, sent) > max_tokens:
            flush_current()
            chunks.extend(_split_by_token_slices(tokenizer, sent, max_tokens))
            continue

        candidate = (" ".join(current + [sent])).strip() if current else sent
        if _token_len(tokenizer, candidate) <= max_tokens:
            current.append(sent)
        else:
            flush_current()
            current.append(sent)

    flush_current()

    safe_chunks: List[str] = []
    for c in chunks:
        if _token_len(tokenizer, c) <= max_tokens:
            safe_chunks.append(c)
        else:
            safe_chunks.extend(_split_by_token_slices(tokenizer, c, max_tokens))
    return safe_chunks


def _aggregate_labels(labels: List[str]) -> str:
    if not labels:
        return "neutral"

    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = [lbl for lbl, c in counts.items() if c == max_count]

    if len(candidates) == 1:
        return candidates[0]

    for p in TIE_BREAK_PRIORITY:
        if p in candidates:
            return p

    return sorted(candidates)[0]


class ModelsInit:
    def __init__(self) -> None:
        self._classification_model = None

    def init_models(self) -> None:
        classification_pipeline = "text-classification"
        classification_model_name = "Sandrro/emotions_classificator_v4"
        logger.info(
            f"Launching classification model {classification_model_name} for {classification_pipeline}"
        )
        self._classification_model = pipeline(
            classification_pipeline,
            model=classification_model_name,
        )


models_initialization = ModelsInit()


def classify_emotion(text: str) -> str:

    labels = classify_emotions([text], batch_size=32, show_progress=False)
    return labels[0] if labels else "neutral"


def classify_emotions(
    texts: Iterable[str],
    *,
    batch_size: int = 32,
    show_progress: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> List[str]:
    model = models_initialization._classification_model
    if model is None:
        raise RuntimeError("Classification model is not initialized. Call init_models() first.")

    texts_list = list(texts)
    total_texts = len(texts_list)
    if not texts_list:
        return []

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Pipeline tokenizer is not available; cannot do token-based chunking.")

    chunk_texts: List[str] = []
    owners: List[int] = []
    chunks_per_text: List[int] = [0] * total_texts

    for i, t in enumerate(texts_list):
        chunks = _chunk_text_by_sentences(tokenizer, str(t), max_tokens=MAX_MODEL_TOKENS)
        if not chunks:
            chunks = [""]
        chunks_per_text[i] = len(chunks)
        for c in chunks:
            chunk_texts.append(c)
            owners.append(i)

    labels_by_text: Dict[int, List[str]] = defaultdict(list)

    done_texts = 0
    processed_chunks_per_text: List[int] = [0] * total_texts

    def mark_chunk_result(owner_idx: int, label: str) -> None:
        nonlocal done_texts
        labels_by_text[owner_idx].append(label)
        processed_chunks_per_text[owner_idx] += 1
        if processed_chunks_per_text[owner_idx] == chunks_per_text[owner_idx]:
            done_texts += 1
            if progress_callback:
                progress_callback(done_texts, total_texts)

    if not show_progress and progress_callback is None:
        results = model(
            chunk_texts,
            batch_size=batch_size,
            truncation=True,
            max_length=MAX_MODEL_TOKENS,
        )
        for owner_idx, item in zip(owners, results):
            mark_chunk_result(owner_idx, item["label"])
        return [_aggregate_labels(labels_by_text[i]) for i in range(total_texts)]

    iterator = range(0, len(chunk_texts), batch_size)
    pbar = tqdm(iterator, desc="Emotions", unit="batch") if show_progress else iterator

    for start in pbar:
        batch = chunk_texts[start : start + batch_size]
        batch_owners = owners[start : start + batch_size]
        results = model(
            batch,
            batch_size=batch_size,
            truncation=True,
            max_length=MAX_MODEL_TOKENS,
        )
        for owner_idx, item in zip(batch_owners, results):
            mark_chunk_result(owner_idx, item["label"])

    return [_aggregate_labels(labels_by_text[i]) for i in range(total_texts)]