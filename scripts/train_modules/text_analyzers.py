"""Текстовые анализаторы для TfidfVectorizer.

Выделено в отдельный модуль, чтобы сериализованная модель (joblib)
ссылалась на лёгкий модуль без зависимостей от mlflow/optuna.
"""

import re
from collections.abc import Callable, Iterable

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def make_tfidf_analyzer(use_stemming: bool) -> str | Callable[[str], Iterable[str]]:
    """Фабрика анализатора для TfidfVectorizer.

    - Если use_stemming=False — возвращает стандартный анализатор "word".
    - Если True — создаёт кастомный анализатор с SnowballStemmer.
    """
    if not use_stemming:
        return "word"

    stemmer = SnowballStemmer("english")
    token_re = re.compile(r"[A-Za-z]+")

    def analyzer(text: str):
        text_clean = re.sub(r"[\u200b\ufeff\u00A0]", " ", text)
        text_clean = re.sub(r"\s+", " ", text_clean).strip()
        tokens = token_re.findall(text_clean.lower())
        tokens = [t for t in tokens if len(t) > 1 and t not in ENGLISH_STOP_WORDS]
        return [stemmer.stem(t) for t in tokens]

    return analyzer
