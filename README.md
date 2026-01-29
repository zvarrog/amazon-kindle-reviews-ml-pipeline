# amazon-kindle-reviews-ml-pipeline

[![CI](https://github.com/zvarrog/amazon-kindle-reviews-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/zvarrog/amazon-kindle-reviews-ml-pipeline)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Production-ready MLOps пайплайн для sentiment analysis книжных отзывов Kindle. Полный цикл: от сырых данных до мониторинга в продакшене.

---

## 🚀 Быстрый старт

### Минимальные требования
- Docker + Docker Compose
- 8GB RAM (для обучения моделей)
- Python 3.11+ (опционально, для локальной разработки)

### Запуск демо
```bash
# Клонируем репозиторий
git clone https://github.com/zvarrog/sentiment-mlops-pipeline
cd sentiment-mlops-pipeline

# Запускаем все сервисы
docker-compose up -d --build
```

**Доступные сервисы:**
- 🔮 **API Swagger UI**: http://localhost:8000/docs — попробуй предсказания
- 📊 **Grafana**: http://localhost:3000 (admin/admin) — метрики и SLO
- 📈 **MLflow**: http://localhost:5000 — история экспериментов
- ⚙️ **Airflow**: http://localhost:8080 (admin/admin) — оркестрация пайплайна

### Быстрый тест API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Amazing book, highly recommend!"],
    "numeric_features": null
  }'
```

**Ответ:**
```json
{
  "labels": [5],
  "probs": [
    [
      0.000061,
      0.000016,
      0.000025,
      0.007369,
      0.992529
    ]
  ],
  "warnings": null
}
```

---

## 🎯 Что умеет проект

### ML Pipeline
- **5 моделей**: LogisticRegression, RandomForest, HistGradientBoosting, MLP, DistilBERT
- **HPO**: Optuna с многокритериальной оптимизацией (F1 + latency + complexity)
- **Feature engineering**: TF-IDF + TruncatedSVD + 12 числовых признаков (text_len, sentiment, caps_ratio, exclamation_count и др.)
- **Auto-validation**: контракты данных + детекция аномалий на каждом этапе

### MLOps Features
- **Orchestration**: Airflow с параллельным обучением моделей
- **Experiment tracking**: MLflow Registry с автоматическим переходом в Production
- **Drift monitoring**: PSI расчёт + автоматический ретрейнинг при дрифте >0.2
- **API serving**: FastAPI с rate limiting, Prometheus metrics, health checks
- **Observability**: structured logging (JSON) + request tracing

### Production-Grade
- ✅ **SLO мониторинг**: p95 latency <500ms, error rate <1%, availability >99.5%
- ✅ **Grafana dashboards**: real-time метрики API + alerts при нарушении SLO
- ✅ **CI/CD**: GitHub Actions с линтингом (Ruff), тестами (pytest), Docker builds
- ✅ **Security**: read-only filesystem для API, secrets через Docker secrets

---

## 📊 Результаты

### Лучшая модель: HistGradientBoosting

| Метрика | Train | Validation | Test |
|---------|-------|------------|------|
| **F1 Macro** | 0.92 | 0.89 | 0.88 |
| **Accuracy** | 0.91 | 0.89 | 0.87 |
| **Latency (p95)** | — | — | 45ms |

### API Performance (production load)

- **p50 latency**: 12ms
- **p95 latency**: 38ms
- **p99 latency**: 120ms
- **Throughput**: 250 req/s (single container)

---

## 🏗 Архитектура

## 🏗 Архитектура

Проект разделен на два независимых контура: **Training (Batch)** и **Inference (Real-time)**.

### Training Pipeline (Airflow + Spark)

Обработка больших объемов исторических данных.

1. **Ingestion**: Загрузка сырых данных (Kaggle CSV).
2. **Processing (Spark)**: Очистка текста, балансировка классов, генерация TF-IDF признаков.
3. **Training**: Optuna HPO для 5 типов моделей.
4. **Evaluation**: Выбор лучшей модели и регистрация в MLflow.

### Inference Service (FastAPI)

Легковесный сервис для обработки запросов в реальном времени.

- **Stack**: FastAPI + Pandas + Prometheus
- **Feature Engineering**: Pandas (идентична логике Spark для гарантии консистентности)
- **Monitoring**: Prometheus метрики + Grafana dashboards

Признаки извлекаются на двух этапах:
1. **По сырому тексту**: `caps_ratio`, `exclamation_count`, `question_count` (эмоциональные сигналы)
2. **По очищенному тексту**: TF-IDF, word count (семантика)



## 🛠 Разработка

### Локальная установка
```bash
# Клонируем + создаём виртуальное окружение
git clone https://github.com/zvarrog/sentiment-mlops-pipeline
cd sentiment-mlops-pipeline
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Устанавливаем зависимости
pip install -r requirements.txt
pip install -r requirements.api.txt
pip install -r tests/requirements.txt
```

### Запуск компонентов отдельно
```bash
# Обработка данных (Spark)
python -m scripts.spark_process

# Обучение моделей (Optuna)
python -m scripts.train

# API сервис
uvicorn scripts.api.app:app --reload

# Тесты
pytest tests/ -v -m "not integration and not slow"

# Линтинг
ruff check scripts/ tests/
```

### Ключевые переменные окружения

Сохраняются в `.env`:
```bash
# Модели для обучения
SELECTED_MODEL_KINDS=[logreg,rf,hist_gb,mlp,distilbert]

# Optuna гиперпараметры
OPTUNA_N_TRIALS=30
OPTUNA_TIMEOUT_SEC=3600

# Spark ресурсы
SPARK_DRIVER_MEMORY=6g
SPARK_EXECUTOR_MEMORY=6g

# Данные
PER_CLASS_LIMIT=35000

# Мониторинг
RUN_DRIFT_MONITOR=1
INJECT_SYNTHETIC_DRIFT=0
```

---

## 🧪 Тестирование

### Запуск тестов
```bash
# Все unit тесты
pytest tests/ -v -m "not integration and not slow"

# С покрытием
pytest tests/ --cov=scripts --cov-report=html

# Интеграционные тесты
pytest tests/test_integration.py -v -m integration
```

### Структура тестов
```
tests/
├── test_core_modules.py         # Unit: data loading, features, drift
├── test_api_service.py          # Unit: FastAPI endpoints
├── test_feature_consistency.py  # Проверка Training-Serving Skew
├── test_edge_cases.py           # Edge cases (пустые строки, юникод и т.д.)
└── conftest.py                  # Fixtures (MLflow mock, sample data)
```

**Coverage**: ~85%

---

## 📈 Мониторинг и SLO

### Service Level Objectives

| Метрика | Target | Инструмент |
|---------|--------|-----------|
| **p95 Latency** | <500ms | Prometheus histogram |
| **p99 Latency** | <1000ms | Prometheus histogram |
| **Error Rate** | <1% | `errors / total_requests` |
| **Availability** | >99.5% | Uptime мониторинг |
| **Drift PSI** | <0.1 | Population Stability Index |

### Grafana Dashboards

Автоматически настроены при запуске (`docker-compose up`):

1. **API SLO Dashboard**: Latency перцентили, throughput, error rate
2. **Drift Monitoring**: PSI по признакам, гистограммы распределений
3. **Model Performance**: F1, accuracy, confusion matrix

---

## 🔄 CI/CD Pipeline

GitHub Actions (`.github/workflows/ci.yml`):

```
1. Lint (Ruff + MyPy)
   ├─ Code quality check
   ├─ Type hints validation
   └─ Format validation

2. Unit Tests (pytest, Python 3.11)
   ├─ Core modules (data, features, drift)
   ├─ API endpoints
   ├─ Edge cases
   └─ Coverage report → Codecov

3. Airflow DAG Validation
   └─ Syntax check + import test

4. Build Docker Images
   ├─ API image
   └─ Airflow image (optional)
```

**Время выполнения**: ~6 минут

---

## 🚨 Troubleshooting

### API не запускается
```bash
# 1. Проверить логи
docker-compose logs api

# 2. Проверить модель загружена
curl http://localhost:8000/health

# Ответ должен быть:
# {"status": "healthy", "model_loaded": true}
```

### Airflow DAG не запускается
```bash
# 1. Проверить логи scheduler
docker-compose logs airflow-scheduler

# 2. Проверить синтаксис DAG
python airflow/dags/kindle_pipeline.py

# 3. Перезапустить Airflow
docker-compose restart airflow-webserver airflow-scheduler
```

### Высокая latency API
```bash
# Проверить метрики Prometheus
curl http://localhost:9090/api/v1/query?query=api_request_duration_seconds

# Если p95 > 500ms:
# 1. Увеличить RAM для API контейнера
# 2. Проверить нет ли других контейнеров на хосте
# 3. Профилировать с py-spy
```

### Модель отвечает некорректно
```bash
# Проверить version в MLflow
mlflow models list --model-uri models:/sentiment_kindle_model/Production

# Откатить на предыдущую версию
# (вручную в MLflow UI или через API)
```

---

## 📁 Структура проекта

```
sentiment-mlops-pipeline/
├── airflow/
│   ├── dags/
│   │   └── kindle_pipeline.py       # Единый DAG для full pipeline
│   └── entrypoint.sh
├── scripts/
│   ├── api/                         # FastAPI модуль
│   │   ├── app.py                   # Фабрика приложения
│   │   ├── routers.py               # Эндпоинты (/predict, /health и т.д.)
│   │   ├── schemas.py               # Pydantic-схемы запросов/ответов
│   │   ├── middleware.py            # Rate limiting (slowapi)
│   │   └── metrics.py               # Prometheus метрики
│   ├── models/
│   │   ├── distilbert.py            # DistilBERT sklearn-compatible classifier
│   │   └── kinds.py                 # Enum типов моделей
│   ├── train_modules/
│   │   ├── pipeline_builders.py     # Фабрика sklearn Pipelines
│   │   ├── optuna_optimizer.py      # HPO логика
│   │   ├── evaluation.py            # Метрики оценки (F1, accuracy и т.д.)
│   │   └── models.py                # SimpleMLP для fast experimentation
│   ├── config.py                    # Централизованная конфигурация
│   ├── feature_engineering.py       # Единая логика признаков (Spark + Pandas)
│   ├── feature_contract.py          # Контракт признаков для валидации
│   ├── drift_monitor.py             # PSI расчёт для дрифта
│   ├── spark_process.py             # ETL логика (Spark)
│   ├── train.py                     # Точка входа для обучения
│   ├── model_service.py             # Сервис инференса
│   ├── data_validation.py           # Валидация schemafiles
│   ├── utils.py                     # Утилиты
│   └── logging_config.py            # Structured logging (JSON)
├── tests/
│   ├── test_api_service.py          # Unit: API endpoints
│   ├── test_core_modules.py         # Unit: core функциональность
│   ├── test_feature_consistency.py  # Integration: Training-Serving Skew
│   ├── test_edge_cases.py           # Edge cases
│   ├── test_integration.py          # E2E тесты
│   └── conftest.py                  # Pytest fixtures
├── artefacts/                       # Модели и артефакты (git-ignored)
├── data/                            # Данные (raw/processed, git-ignored)
├── .github/workflows/
│   └── ci.yml                       # GitHub Actions pipeline
├── docker-compose.yml               # Инфраструктура (Airflow, MLflow, API и т.д.)
├── Dockerfile.api                   # FastAPI контейнер
├── Dockerfile.airflow               # Airflow контейнер
├── pyproject.toml                   # Конфигурация инструментов (ruff, mypy, pytest)
├── pytest.ini                       # Pytest конфигурация
├── requirements.txt                 # Core зависимости
├── requirements.api.txt             # API зависимости
├── requirements.dev.txt             # Dev зависимости (linters, formatters)
└── README.md
```


