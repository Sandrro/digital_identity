from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class AxisDefinition:
    name: str
    left_terms: Sequence[str]
    right_terms: Sequence[str]


def parse_axis_lines(axis_lines: str) -> List[AxisDefinition]:
    axes: List[AxisDefinition] = []
    for raw_line in axis_lines.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3:
            raise ValueError(
                "Каждая ось должна быть задана как 'Название | левые слова | правые слова'."
            )
        name, left_raw, right_raw = parts[:3]
        left_terms = _split_terms(left_raw)
        right_terms = _split_terms(right_raw)
        if not left_terms or not right_terms:
            raise ValueError("Для каждой оси нужны оба набора слов.")
        axes.append(
            AxisDefinition(name=name, left_terms=left_terms, right_terms=right_terms)
        )
    return axes


def score_texts_on_axes(
    texts: Iterable[str],
    axes: Sequence[AxisDefinition],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> pd.DataFrame:
    texts_list = [text if text is not None else "" for text in texts]
    if not axes:
        return pd.DataFrame(index=range(len(texts_list)))

    model = SentenceTransformer(model_name)
    text_embeddings = model.encode(
        texts_list, normalize_embeddings=True, batch_size=batch_size
    )

    left_texts = [" ".join(axis.left_terms) for axis in axes]
    right_texts = [" ".join(axis.right_terms) for axis in axes]
    left_embeddings = model.encode(
        left_texts, normalize_embeddings=True, batch_size=batch_size
    )
    right_embeddings = model.encode(
        right_texts, normalize_embeddings=True, batch_size=batch_size
    )

    similarity_left = np.matmul(text_embeddings, left_embeddings.T)
    similarity_right = np.matmul(text_embeddings, right_embeddings.T)

    exp_left = np.exp(similarity_left)
    exp_right = np.exp(similarity_right)
    prob_right = exp_right / (exp_left + exp_right)
    scores = 1 + 9 * prob_right

    axis_names = [axis.name for axis in axes]
    return pd.DataFrame(scores, columns=axis_names)


def build_radar_chart(average_scores: pd.Series, *, title: str = "") -> go.Figure:
    labels = list(average_scores.index)
    values = average_scores.values.tolist()
    if labels:
        labels += labels[:1]
        values += values[:1]

    fig = go.Figure(
        data=[
            go.Scatterpolar(r=values, theta=labels, fill="toself", name="Среднее")
        ]
    )
    fig.update_layout(
        title=title or "Семантическая лепестковая диаграмма",
        polar=dict(radialaxis=dict(visible=True, range=[1, 10])),
        showlegend=False,
    )
    return fig


def build_semantic_map(
    scores: pd.DataFrame,
    texts: Sequence[str],
    *,
    axis_x: str,
    axis_y: str,
    title: str = "",
) -> px.scatter:
    data = pd.DataFrame(
        {
            axis_x: scores[axis_x],
            axis_y: scores[axis_y],
            "text": texts,
        }
    )
    fig = px.scatter(
        data,
        x=axis_x,
        y=axis_y,
        hover_data={"text": True},
        title=title or "Семантическая карта",
    )
    fig.update_layout(
        xaxis=dict(range=[1, 10]),
        yaxis=dict(range=[1, 10]),
    )
    return fig


def _split_terms(raw: str) -> List[str]:
    return [term.strip() for term in raw.split(",") if term.strip()]
