import logging
from typing import Callable, Iterable, List

from transformers import pipeline
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


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
            classification_pipeline, model=classification_model_name
        )


models_initialization = ModelsInit()


def classify_emotion(text: str) -> str:
    """
    Функция для получения предсказания по тексту с помощью модели классификации.
    """
    model = models_initialization._classification_model
    if model is None:
        raise RuntimeError("Classification model is not initialized. Call init_models() first.")
    result = model(text)
    return result[0]["label"]


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
    total = len(texts_list)
    if not texts_list:
        return []
    if not show_progress and progress_callback is None:
        results = model(texts_list, batch_size=batch_size)
        return [item["label"] for item in results]

    labels: List[str] = []
    steps = range(0, total, batch_size)
    iterator = (
        tqdm(steps, desc="Emotions", unit="batch")
        if show_progress
        else steps
    )
    for start in iterator:
        batch = texts_list[start : start + batch_size]
        results = model(batch, batch_size=batch_size)
        labels.extend(item["label"] for item in results)
        if progress_callback:
            progress_callback(len(labels), total)
    return labels
