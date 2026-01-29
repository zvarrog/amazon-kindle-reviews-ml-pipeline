"""Единый модуль генерации фичей для API, Spark и обучения.

Обеспечивает консистентность признаков между этапами обучения и инференса (Serving).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import cast

import pandas as pd
from textblob import TextBlob

from scripts.logging_config import get_logger

log = get_logger(__name__)


def clean_text(text: str | None) -> str:
    """Очищает текст для NLP моделей.

    1. Нижний регистр.
    2. Удаление URL.
    3. Оставление только латиницы и пробелов.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_sentiment(text: str | None) -> float:
    """Расчёт полярности текста через TextBlob [-1.0, 1.0]."""
    if not text or not isinstance(text, str) or len(text.strip()) < 3:
        return 0.0
    try:
        blob = TextBlob(text)
        return float(round(blob.sentiment.polarity, 4))
    except (AttributeError, ValueError, TypeError) as e:
        log.debug("Ошибка sentiment: %s", e)
        return 0.0


def extract_text_features(text: str) -> dict[str, float]:
    """Извлечение статистики из сырого текста."""
    if not text or not isinstance(text, str):
        return dict.fromkeys(
            [
                "text_len",
                "word_count",
                "kindle_freq",
                "exclamation_count",
                "caps_ratio",
                "question_count",
                "avg_word_length",
            ],
            0.0,
        )

    text_len = float(len(text))
    words = text.split()
    word_count = float(len(words))

    return {
        "text_len": text_len,
        "word_count": word_count,
        "kindle_freq": float(text.lower().count("kindle")),
        "exclamation_count": float(text.count("!")),
        "question_count": float(text.count("?")),
        "caps_ratio": sum(1 for c in text if c.isupper()) / text_len if text_len else 0.0,
        "avg_word_length": text_len / word_count if word_count else 0.0,
    }


def extract_pandas_features(series: pd.Series) -> pd.DataFrame:
    """Высокопроизводительное извлечение признаков для батч-обработки."""
    s = series.fillna("")

    text_len = s.str.len().astype(float)
    word_count = s.str.split().str.len().fillna(0).astype(float)

    res = pd.DataFrame(
        {
            "text_len": text_len,
            "word_count": word_count,
            "kindle_freq": s.str.count("kindle", flags=re.IGNORECASE).astype(float),
            "exclamation_count": s.str.count("!").astype(float),
            "question_count": s.str.count(r"\?").astype(float),
            "caps_ratio": s.apply(
                lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) else 0.0
            ).astype(float),
            "avg_word_length": (text_len / word_count.replace(0, 1)).astype(float),
        }
    )
    return res


def transform_features(
    texts: Iterable[str],
    numeric_features: dict[str, list[float]] | None,
    expected_numeric_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Трансформация входных данных в DataFrame для предикта.

    Args:
        texts: Список текстов отзывов.
        numeric_features: Дополнительные числовые фичи (если есть).
        expected_numeric_cols: Список фичей, которые ожидает обученная модель.
    """
    df = pd.DataFrame({"reviewText": list(texts)})
    raw_texts = df["reviewText"].fillna("")

    stats_df = extract_pandas_features(cast(pd.Series, raw_texts))

    df["reviewText"] = raw_texts.apply(clean_text)

    df["sentiment"] = df["reviewText"].apply(calculate_sentiment)

    df = pd.concat([df, stats_df], axis=1)

    errors: list[str] = []
    if numeric_features:
        for col in expected_numeric_cols:
            if col in numeric_features:
                vals = numeric_features[col]
                if len(vals) == len(df):
                    # pd.to_numeric может вернуть ndarray если входное значение ndarray,
                    numeric_vals = pd.to_numeric(pd.Series(vals), errors="coerce").fillna(0.0)
                    df[col] = numeric_vals.values
                else:
                    errors.append(f"Mismatch length for {col}: {len(vals)} != {len(df)}")

    missing_cols = [col for col in expected_numeric_cols if col not in df.columns]
    if missing_cols:
        log.warning("Отсутствуют колонки, заполняю нулями: %s", missing_cols)
        for col in missing_cols:
            df[col] = 0.0

    return df, errors


# Spark UDFs вынесены в функции-фабрики для избежания проблем с сериализацией
def get_spark_clean_udf():
    """Создаёт Spark UDF для очистки текста."""
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    return udf(lambda t: clean_text(t), StringType())


def get_spark_feature_extraction_udf():
    """Создаёт Spark UDF для извлечения текстовых характеристик (caps, signs etc)."""
    from pyspark.sql.functions import udf
    from pyspark.sql.types import FloatType, MapType, StringType

    return udf(lambda t: extract_text_features(t) if t else {}, MapType(StringType(), FloatType()))
