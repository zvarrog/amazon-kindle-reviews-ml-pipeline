#!/bin/bash
set -e

echo "=== Инициализация Airflow ==="
mkdir -p /opt/airflow/data/processed /opt/airflow/data/raw

chown -R 50000:0 /opt/airflow/data 2>/dev/null || chmod -R 777 /opt/airflow/data 2>/dev/null || true

rm -f /opt/airflow/airflow-webserver*.pid
echo "Миграция базы данных..."
airflow db migrate

if ! airflow users list | grep -q "admin@example.com"; then
    echo "Создание admin пользователя..."
    airflow users create \
        --username admin \
        --password admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com
fi

echo "Запуск Airflow webserver..."
airflow webserver -p 8080 &
WEBSERVER_PID=$!

echo "Ожидание запуска Airflow UI..."
for i in {1..30}; do
    if curl -sSf http://localhost:8080/health >/dev/null 2>&1; then
        echo "Airflow UI доступен: http://localhost:8080 (admin/admin)"
        break
    fi
    sleep 2
done

echo "Запуск Airflow scheduler..."
exec airflow scheduler
