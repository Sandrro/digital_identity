import logging
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
from transformers import pipeline
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

MAX_MODEL_TOKENS = 512
TIE_BREAK_PRIORITY = ["negative", "positive", "neutral"]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_into_sentences(text: str) -> List[str]:
    """
    Dependency-free sentence splitting.
    Good enough for typical RU/EN punctuation.
    """
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def _split_by_token_slices(tokenizer, text: str, max_tokens: int) -> List[str]:
    """
    If a single piece is still > max_tokens, slice by tokens (no special tokens).
    Guarantees each returned chunk will be <= max_tokens with special tokens added.
    """
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

    safe_chunks: List[str] = []
    for c in chunks:
        if _token_len(tokenizer, c) <= max_tokens:
            safe_chunks.append(c)
        else:
            safe_chunks.extend(_split_by_token_slices(tokenizer, c, max_tokens))
    return safe_chunks


def _chunk_text_by_sentences(tokenizer, text: str, max_tokens: int = MAX_MODEL_TOKENS) -> List[str]:
    """
    1) Split into sentences
    2) Pack sentences into chunks not exceeding max_tokens
    3) If a sentence is too long, slice by tokens
    """
    text = _normalize_ws(str(text))
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
    """
    Majority vote. Tie-break: negative > positive > neutral.
    If none of those present among tied candidates: deterministic by alphabet.
    """
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


def _resolve_pipeline_device(p) -> torch.device:
    """
    HuggingFace pipeline may expose device as:
    - int (e.g. -1, 0, 1)
    - torch.device
    - str ('cpu', 'cuda', 'cuda:0')
    - None
    We normalize to torch.device.
    """
    dev = getattr(p, "device", None)

    if isinstance(dev, torch.device):
        return dev
    if dev is None:
        return torch.device("cpu")
    if isinstance(dev, str):
        try:
            return torch.device(dev)
        except Exception:
            return torch.device("cpu")

    if isinstance(dev, int):
        if dev < 0:
            return torch.device("cpu")
        return torch.device(f"cuda:{dev}")

    try:
        return torch.device(str(dev))
    except Exception:
        return torch.device("cpu")


def _predict_labels_strict(p, texts: List[str], *, max_length: int = MAX_MODEL_TOKENS) -> List[str]:
    """
    Strict max_length guarantee by manual tokenization + direct model forward.
    This avoids cases where pipeline call doesn't apply truncation as expected.
    """
    tokenizer = getattr(p, "tokenizer", None)
    mdl = getattr(p, "model", None)
    if tokenizer is None or mdl is None:
        raise RuntimeError("Pipeline must have tokenizer and model.")

    device = _resolve_pipeline_device(p)

    batch = [("" if t is None else str(t)) for t in texts]

    enc = tokenizer(
        batch,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    mdl = mdl.to(device)
    mdl.eval()

    with torch.no_grad():
        out = mdl(**enc)
        logits = out.logits
        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()

    id2label = getattr(mdl.config, "id2label", None) or {}
    labels = [id2label.get(i, str(i)) for i in pred_ids]
    return labels


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
    """
    Single-text wrapper (with chunking + collapse).
    """
    labels = classify_emotions([text], batch_size=32, show_progress=False)
    return labels[0] if labels else "neutral"


def classify_emotions(
    texts: Iterable[str],
    *,
    batch_size: int = 32,
    show_progress: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> List[str]:
    p = models_initialization._classification_model
    if p is None:
        raise RuntimeError("Classification model is not initialized. Call init_models() first.")

    texts_list = list(texts)
    total_texts = len(texts_list)
    if not texts_list:
        return []

    tokenizer = getattr(p, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Pipeline tokenizer is not available; cannot do token-based chunking.")

    chunk_texts: List[str] = []
    owners: List[int] = []
    chunks_per_text: List[int] = [0] * total_texts

    for i, t in enumerate(texts_list):
        chunks = _chunk_text_by_sentences(tokenizer, t, max_tokens=MAX_MODEL_TOKENS)
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

    iterator = range(0, len(chunk_texts), batch_size)
    pbar = tqdm(iterator, desc="Emotions", unit="batch") if show_progress else iterator

    for start in pbar:
        batch = chunk_texts[start : start + batch_size]
        batch_owners = owners[start : start + batch_size]

        batch_labels = _predict_labels_strict(p, batch, max_length=MAX_MODEL_TOKENS)

        for owner_idx, lbl in zip(batch_owners, batch_labels):
            mark_chunk_result(owner_idx, lbl)

    return [_aggregate_labels(labels_by_text[i]) for i in range(total_texts)]
