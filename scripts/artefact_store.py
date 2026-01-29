"""Унифицированный интерфейс для работы с артефактами.

Решает проблемы:
* Разные способы сохранения JSON/CSV/Parquet
* Отсутствие atomic writes
* Сложность тестирования I/O операций

Упрощённая версия без Protocol: достаточно конкретной реализации
`LocalArtifactStore`, так как в проекте нет альтернативных хранилищ
(S3, GCS и т.п.) и не требуется структурная проверка типов.
"""

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from scripts.logging_config import get_logger

log = get_logger(__name__)


class LocalArtifactStore:
    """Локальное файловое хранилище с atomic writes."""

    def _atomic_write(self, path: Path, write_func: Any) -> None:
        """Вспомогательный метод для атомарной записи через временный файл."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            write_func(temp_path)
            os.replace(temp_path, path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def save_json(self, path: Path, data: dict[str, Any], indent: int = 2, **kwargs: Any) -> None:
        """Атомарная запись JSON."""
        self._atomic_write(
            path,
            lambda p: json.dump(
                data, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=indent, **kwargs
            ),
        )
        log.debug("JSON артефакт сохранен: %s", path)

    def save_csv(self, path: Path, df: pd.DataFrame, index: bool = False, **kwargs: Any) -> None:
        """Атомарная запись CSV."""
        self._atomic_write(path, lambda p: df.to_csv(p, index=index, **kwargs))
        log.debug("CSV артефакт сохранен: %s", path)

    def save_parquet(self, path: Path, df: pd.DataFrame, **kwargs: Any) -> None:
        """Атомарная запись Parquet."""
        self._atomic_write(path, lambda p: df.to_parquet(p, **kwargs))
        log.debug("Parquet артефакт сохранен: %s", path)

    def save_text(self, path: Path, content: str) -> None:
        """Атомарная запись текстового файла."""

        def _write(p: Path):
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)

        self._atomic_write(path, _write)
        log.debug("Текстовый артефакт сохранен: %s", path)

    def save_model(self, path: Path, model: Any) -> None:
        """Сохраняет модель атомарно с использованием joblib."""
        self._atomic_write(path, lambda p: joblib.dump(model, p))
        log.info("Модель сохранена: %s", path)


artefact_store = LocalArtifactStore()
