"""Unit tests для API service (FastAPI).

Фокус: корректность схемы запросов/ответов и устойчивость без реальных артефактов.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


class DummyModel:
    """Заглушечная модель для тестов."""

    def predict(self, X):
        # Поддерживаем как pd.Series, так и pd.DataFrame
        try:
            n = len(X)
        except TypeError:
            n = 1
        # Возвращаем нули
        return np.zeros(n, dtype=int)

    def predict_proba(self, X):
        try:
            n = len(X)
        except TypeError:
            n = 1
        probs0 = np.full(n, 0.6)
        probs1 = np.full(n, 0.4)
        return np.column_stack([probs0, probs1])


@pytest.fixture(scope="module")
def mock_model():
    """Создаёт простую модель для тестов."""
    return DummyModel()


@pytest.fixture(scope="module")
def mock_feature_contract():
    """Создаёт mock feature contract."""
    contract = MagicMock()
    contract.validate_input_data.return_value = {}
    contract.required_text_columns = ["reviewText"]
    contract.expected_numeric_columns = ["text_len", "word_count"]
    contract.get_feature_info.return_value = {
        "required_text_columns": ["reviewText"],
        "expected_numeric_columns": ["text_len", "word_count"],
        "baseline_available": False,
    }
    return contract


@pytest.fixture(scope="module")
def test_client(mock_model, mock_feature_contract, tmp_path_factory):
    """Создаёт тестовый клиент FastAPI с моками и пропуском загрузки артефактов."""
    tmp_model = tmp_path_factory.mktemp("model") / "best_model.joblib"
    tmp_model.write_bytes(b"fake")

    with (
        patch("scripts.config.BEST_MODEL_PATH", tmp_model),
        patch("scripts.model_service.BEST_MODEL_PATH", tmp_model),
        patch("scripts.model_service.joblib.load", return_value=mock_model),
        patch("scripts.model_service.FeatureContract") as mock_contract_cls,
    ):
        mock_contract_cls.from_model_artifacts.return_value = mock_feature_contract

        from scripts.api.app import create_app

        app = create_app(defer_artifacts=False)
        # Используем context manager для запуска lifespan
        with TestClient(app) as client:
            app.state.META = {"best_model": "logreg"}
            app.state.NUMERIC_DEFAULTS = {
                "text_len": {"mean": 10.0},
                "word_count": {"mean": 2.0},
            }
            app.state.FEATURE_CONTRACT = mock_feature_contract
            yield client


class TestAPIServiceHealthCheck:
    """Тесты для health check эндпоинта."""

    def test_health_check_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data


class TestAPISinglePrediction:
    """Тесты для /predict эндпоинта."""

    def test_predict_with_valid_input(self, test_client):
        payload = {
            "texts": ["great product"],
            "numeric_features": {"text_len": [13.0], "word_count": [2.0]},
        }
        resp = test_client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "labels" in data

    def test_predict_with_missing_required_field(self, test_client):
        payload = {"numeric_features": {"text_len": [13.0], "word_count": [2.0]}}
        resp = test_client.post("/predict", json=payload)
        # Pydantic схемы вернут 422 при отсутствии обязательного поля texts
        assert resp.status_code == 422

    def test_predict_with_invalid_type(self, test_client):
        payload = {"texts": "not_a_list"}
        resp = test_client.post("/predict", json=payload)
        assert resp.status_code == 422


class TestAPIBatchPrediction:
    """Тесты для /batch_predict эндпоинта."""

    def test_batch_predict_with_valid_input(self, test_client):
        payload = {
            "data": [
                {"reviewText": "great product", "text_len": 13.0, "word_count": 2.0},
                {"reviewText": "bad quality", "text_len": 11.0, "word_count": 2.0},
            ]
        }
        resp = test_client.post("/batch_predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data and len(data["predictions"]) == 2

    def test_batch_predict_with_empty_list(self, test_client):
        payload = {"data": []}
        resp = test_client.post("/batch_predict", json=payload)
        # Pydantic min_length=1 возвращает 422 Unprocessable Entity
        assert resp.status_code == 422

    def test_batch_predict_exceeds_limit(self, test_client):
        payload = {"data": [{"reviewText": "t"}] * 1001}
        # Pydantic max_length возвращает 422 Unprocessable Entity
        resp = test_client.post("/batch_predict", json=payload)
        assert resp.status_code == 422


class TestAPIMetrics:
    """Smoke tests для Prometheus metrics."""

    def test_metrics_endpoint_exists(self, test_client):
        """GET /metrics возвращает 200."""
        response = test_client.get("/metrics")
        assert response.status_code == 200


class TestAPIErrorPaths:
    """Тесты для error paths API."""

    def test_ready_returns_503_when_model_not_loaded(self):
        """GET /ready возвращает 503 когда модель не загружена."""
        with patch("scripts.model_service.BEST_MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False

            from scripts.api.app import create_app

            app = create_app(defer_artifacts=True)
            with TestClient(app) as client:
                resp = client.get("/ready")
                assert resp.status_code == 503
                data = resp.json()
                assert data["status"] == "not_ready"
                assert data["model_loaded"] is False

    def test_predict_returns_503_when_model_not_loaded(self):
        """POST /predict возвращает 503 когда модель не загружена (ожидается retry)."""
        with patch("scripts.model_service.BEST_MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False

            from scripts.api.app import create_app

            app = create_app(defer_artifacts=True)
            with TestClient(app) as client:
                payload = {"texts": ["test"]}
                resp = client.post("/predict", json=payload)
                # Модель не загружена — возвращаем 503 Service Unavailable
                assert resp.status_code in (500, 503)

    def test_predict_with_empty_texts_list(self, test_client):
        """POST /predict с пустым списком текстов возвращает 422."""
        payload = {"texts": []}
        resp = test_client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_with_text_exceeding_max_length(self, test_client):
        """POST /predict с текстом превышающим MAX_TEXT_LENGTH возвращает 422."""
        # MAX_TEXT_LENGTH по умолчанию 2000
        long_text = "a" * 3000
        payload = {"texts": [long_text]}
        resp = test_client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_metadata_returns_valid_structure(self, test_client):
        """GET /metadata возвращает правильную структуру."""
        resp = test_client.get("/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_info" in data
        assert "feature_contract" in data
        assert "health" in data
