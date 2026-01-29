"""Параметризованный DAG для обработки Kindle reviews."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from airflow.decorators import dag, task
from airflow.models.param import Param

try:
    from scripts.config import (
        DATA_PATHS,
        DRIFT_ARTEFACTS_DIR,
        PROCESSED_DATA_DIR,
        SELECTED_MODEL_KINDS,
    )
    from scripts.models.kinds import ModelKind
except ImportError:
    logging.warning("Модуль scripts не найден. Убедитесь, что PYTHONPATH настроен.")
    SELECTED_MODEL_KINDS = []


log = logging.getLogger(__name__)


@dag(
    dag_id="amazon_kindle_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["sentiment", "ml", "amazon"],
    params={
        "parallel": Param(False, type="boolean"),
        "force_download": Param(False, type="boolean"),
        "force_process": Param(False, type="boolean"),
        "force_train": Param(False, type="boolean"),
        "run_data_validation": Param(True, type="boolean"),
        "inject_synthetic_drift": Param(False, type="boolean"),
        "run_drift_monitor": Param(False, type="boolean"),
    },
)
def kindle_pipeline():
    """Главный DAG для пайплайна обучения и обработки."""

    @task
    def download_task(params: dict[str, Any] = None) -> str:
        from scripts.download import main as download_main

        force = params.get("force_download", False)
        log.info(f"Загрузка данных (force={force})")
        return str(download_main(force=force))

    @task
    def validate_task(prev_result: str, params: dict[str, Any] = None) -> None:
        from scripts.data_validation import main as validate_main

        if not params.get("run_data_validation", True):
            log.info("Валидация пропущена")
            return

        log.info("Валидация данных")
        if not validate_main():
            raise ValueError("Ошибки валидации данных")

    @task
    def inject_drift_task(prev_result: None, params: dict[str, Any] = None) -> dict[str, Any]:
        """Инъекция дрейфа."""
        if not params.get("inject_synthetic_drift", False):
            return {"status": "skipped"}

        from scripts.drift_injection import main as inject_main

        log.info("Инъекция дрейфа")
        result = inject_main()
        if result.get("status") == "error":
            raise RuntimeError(f"Ошибка инъекции дрейфа: {result.get('message')}")
        return result

    @task
    def process_task(prev_result: Any, params: dict[str, Any] = None) -> dict[str, str]:
        from scripts.spark_process import process_data

        force = params.get("force_process", False)
        log.info(f"Обработка данных (force={force})")
        process_data(force=force)

        return {
            "train": str(DATA_PATHS.train),
            "val": str(DATA_PATHS.val),
            "test": str(DATA_PATHS.test),
        }

    @task
    def drift_monitor_task(paths: dict[str, str], params: dict[str, Any] = None) -> dict[str, Any]:
        from scripts.drift_monitor import run_drift_monitor

        if not params.get("run_drift_monitor", False):
            return {"status": "skipped"}

        test_parquet = Path(paths.get("test", "unknown"))
        if not test_parquet.exists():
            test_parquet = PROCESSED_DATA_DIR / "test.parquet"

        log.info(f"Запуск мониторинга дрейфа для {test_parquet}")
        if not test_parquet.exists():
            raise FileNotFoundError(f"Файл {test_parquet} не найден")

        report = run_drift_monitor(
            str(test_parquet),
            threshold=0.2,
            save=True,
            out_dir=Path(DRIFT_ARTEFACTS_DIR),
        )

        drifted = [r.get("feature") for r in report if r.get("drift")]
        if drifted:
            log.warning(f"Обнаружен дрифт: {drifted}")

        return {"count": len(report), "drifted": drifted}

    @task.branch
    def check_mode(params: dict[str, Any] = None) -> str | list[str]:
        if params.get("parallel", False):
            return "train_parallel"
        return "train_standard_task"

    @task
    def train_standard_task(params: dict[str, Any] = None) -> str:
        from scripts.train import run as train_run

        force = params.get("force_train", False)
        log.info(f"Обучение (standard, force={force})")
        train_run(force=force)
        return "success"

    @task
    def train_parallel(model_kind_str: str, params: dict[str, Any] = None) -> dict[str, Any]:
        from scripts.train import run as train_run

        force = params.get("force_train", False)
        log.info(f"Обучение модели: {model_kind_str}")

        train_run(
            force=force,
            selected_models=[ModelKind(model_kind_str)],
        )
        return {"model": model_kind_str, "status": "completed"}

    @task
    def select_best_task(results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            log.warning("Нет результатов параллельного обучения")
            return {"status": "empty"}

        log.info(f"Обработка результатов: {len(results)}")
        return {"status": "complete", "models": [r["model"] for r in results]}

    dl = download_task()
    val = validate_task(dl)
    inj = inject_drift_task(val)
    proc_paths = process_task(inj)
    mon = drift_monitor_task(proc_paths)

    mode = check_mode()
    mon >> mode

    std_train = train_standard_task()

    kind_values = [mk.value for mk in SELECTED_MODEL_KINDS] if SELECTED_MODEL_KINDS else []
    par_train = train_parallel.expand(model_kind_str=kind_values)

    select_best_task(par_train)

    mode >> [std_train, par_train]


dag = kindle_pipeline()
