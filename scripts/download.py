"""Загрузка данных с Kaggle.

Используется Kaggle API для загрузки и распаковки датасета kindle-reviews.
"""

from pathlib import Path

from scripts.config import (
    CSV_NAME,
    FORCE_DOWNLOAD,
    KAGGLE_DATASET,
    RAW_DATA_DIR,
)
from scripts.logging_config import get_logger

log = get_logger(__name__)


def download_dataset(
    dataset_id: str,
    dest_dir: Path,
    unzip: bool = True,
) -> None:
    """Скачивание датасета через Kaggle API.

    Импорт kaggle вынесен внутрь, чтобы избежать побочных эффектов
    (поиск конфигов) при простом импорте модуля.
    """
    try:
        import kaggle
    except ImportError:
        log.error("Библиотека 'kaggle' не установлена. Установите её через pip install kaggle.")
        raise

    log.info("Запрос к Kaggle API для датасета: %s", dataset_id)

    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        kaggle.api.dataset_download_files(dataset_id, path=str(dest_dir), unzip=unzip, quiet=False)
    except Exception as e:
        log.error("Ошибка Kaggle API: %s. Убедитесь, что kaggle.json настроен.", e)
        raise


def main(force: bool = False, dataset_id: str | None = None) -> Path:
    """Точка входа для загрузки данных.

    Args:
        force: Принудительная перезагрузка, даже если файл существует.
        dataset_id: Переопределение ID датасета (опционально).

    Returns:
        Path: Абсолютный путь к загруженному CSV.
    """
    target_dataset = dataset_id or KAGGLE_DATASET
    csv_path = RAW_DATA_DIR / CSV_NAME

    should_download = force or FORCE_DOWNLOAD or not csv_path.exists()

    if not should_download:
        log.info("Файл %s уже существует. Пропускаем загрузку.", csv_path)
        return csv_path.resolve()

    log.info("Начало загрузки данных...")
    download_dataset(target_dataset, RAW_DATA_DIR)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Файл {CSV_NAME} не найден в {RAW_DATA_DIR} после загрузки. "
            "Возможно, структура датасета изменилась."
        )

    log.info("Данные готовы: %s", csv_path.resolve())
    return csv_path.resolve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Загрузка данных с Kaggle")
    parser.add_argument("--force", action="store_true", help="Принудительная перезагрузка")
    args = parser.parse_args()

    try:
        main(force=args.force)
    except Exception as err:
        log.critical("Загрузка данных прервана: %s", err)
        import sys

        sys.exit(1)
