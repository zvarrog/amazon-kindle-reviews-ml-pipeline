"""Инъекция синтетического дрейфа для тестирования системы мониторинга."""

from pathlib import Path

import pandas as pd

from scripts.artefact_store import artefact_store
from scripts.config import DATA_PATHS, INJECT_SYNTHETIC_DRIFT, NUMERIC_COLS
from scripts.logging_config import get_logger

log = get_logger(__name__)


def inject_synthetic_drift(
    test_data_path: Path | str | None = None,
) -> dict[str, str | list[str]]:
    """Инъектирует синтетический дрейф в тестовые данные.

    Args:
        test_data_path: Путь к тестовым данным. Если None, используется стандартный путь.

    Returns:
        dict: Результат операции с ключами 'status', 'message', 'changed_columns'
    """
    if not INJECT_SYNTHETIC_DRIFT:
        log.info("Инъекция дрейфа отключена: INJECT_SYNTHETIC_DRIFT=0")
        return {
            "status": "skipped",
            "message": "INJECT_SYNTHETIC_DRIFT=0",
            "changed_columns": [],
        }

    test_path = DATA_PATHS.test if test_data_path is None else Path(test_data_path)

    if not test_path.exists():
        error_msg = f"Тестовые данные не найдены: {test_path}"
        log.error(error_msg)
        return {"status": "error", "message": error_msg, "changed_columns": []}

    try:
        df = pd.read_parquet(test_path)

        changed_columns = _apply_synthetic_drift(df)

        if changed_columns:
            artefact_store.save_parquet(test_path, df)

            success_msg = f"Синтетический дрейф применён к колонкам: {', '.join(changed_columns)}"
            log.warning(success_msg)
            return {
                "status": "success",
                "message": success_msg,
                "changed_columns": changed_columns,
            }
        else:
            warning_msg = "Подходящих числовых колонок для инъекции дрейфа не найдено"
            log.warning(warning_msg)
            return {
                "status": "no_changes",
                "message": warning_msg,
                "changed_columns": [],
            }

    except (OSError, ValueError) as e:
        error_msg = f"Ошибка при инъекции синтетического дрейфа: {e}"
        log.error(error_msg)
        return {"status": "error", "message": error_msg, "changed_columns": []}


def _apply_synthetic_drift(df: pd.DataFrame) -> list[str]:
    """Применяет синтетический дрейф к числовым колонкам.

    Args:
        df: DataFrame для модификации (изменяется in-place)

    Returns:
        list: Список изменённых колонок
    """
    candidate_columns = NUMERIC_COLS

    changed_columns = []

    for col in candidate_columns:
        if col in df.columns:
            try:
                df[col] = df[col] * 1.5 + 10.0

                changed_columns.append(col)
                log.debug("Применён дрейф к колонке '%s'", col)

            except (ValueError, TypeError) as e:
                log.warning("Не удалось применить дрейф к колонке '%s': %s", col, e)

    return changed_columns


def main() -> dict[str, str | list[str]]:
    """Главная функция для запуска инъекции дрейфа из командной строки или DAG."""
    result = inject_synthetic_drift()
    log.info("Результат инъекции: %s - %s", result["status"], result["message"])
    return result


if __name__ == "__main__":
    main()
