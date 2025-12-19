import logging
from typing import Iterable, List

from transformers import pipeline

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


def classify_emotions(texts: Iterable[str]) -> List[str]:
    model = models_initialization._classification_model
    if model is None:
        raise RuntimeError("Classification model is not initialized. Call init_models() first.")
    results = model(list(texts))
    return [item["label"] for item in results]
