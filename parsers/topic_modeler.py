from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

try:
    from hdbscan import HDBSCAN
except ImportError:  # pragma: no cover - optional dependency
    HDBSCAN = None


@dataclass
class TopicModelResult:
    model: BERTopic
    topics: List[int]
    probabilities: List[List[float]] | None


def train_topic_model(
    texts: Iterable[str],
    *,
    language: str = "multilingual",
    min_topic_size: int = 10,
    nr_topics: int | None = None,
    embedding_model: str | None = None,
    stop_words: str | list[str] | None = None,
    cluster_method: str = "hdbscan",
    n_clusters: int | None = None,
    min_samples: int | None = None,
    reduce_frequent_words: bool = True,
    calculate_probabilities: bool = False,
    verbose: bool = True,
) -> TopicModelResult:
    vectorizer_model = CountVectorizer(stop_words=stop_words)
    cluster_method = cluster_method.lower()
    if cluster_method == "kmeans":
        n_clusters = n_clusters or max(2, min_topic_size)
        hdbscan_model = KMeans(n_clusters=n_clusters, random_state=42)
    elif cluster_method == "hdbscan":
        if HDBSCAN is None:
            raise RuntimeError(
                "HDBSCAN is not installed. Install hdbscan or choose kmeans."
            )
        min_samples = min_samples if min_samples is not None else max(1, min_topic_size // 2)
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=min_samples,
            prediction_data=True,
        )
    else:
        raise ValueError("cluster_method must be 'hdbscan' or 'kmeans'")
    model = BERTopic(
        language=language,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        embedding_model=embedding_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=reduce_frequent_words),
        calculate_probabilities=calculate_probabilities,
        verbose=verbose,
    )
    topics, probabilities = model.fit_transform(list(texts))
    return TopicModelResult(model=model, topics=topics, probabilities=probabilities)


def topic_keywords(model: BERTopic, topic_id: int, top_n: int = 5) -> str:
    if topic_id == -1:
        return ""
    words = model.get_topic(topic_id) or []
    return ", ".join(word for word, _ in words[:top_n])


def build_topic_keywords_map(
    model: BERTopic, topics: Iterable[int], top_n: int = 5
) -> dict[int, str]:
    return {topic_id: topic_keywords(model, topic_id, top_n) for topic_id in set(topics)}


def attach_topics(
    data,
    topics: Iterable[int],
    model: BERTopic,
    *,
    topic_column: str = "topic_id",
    keywords_column: str = "topic_keywords",
    top_n: int = 5,
):
    result = data.copy()
    topics_list = list(topics)
    keywords_map = build_topic_keywords_map(model, topics_list, top_n=top_n)
    result[topic_column] = topics_list
    result[keywords_column] = [keywords_map.get(topic_id, "") for topic_id in topics_list]
    return result
