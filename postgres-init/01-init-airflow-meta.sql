SELECT 'CREATE DATABASE airflow_meta'
WHERE NOT EXISTS (
        SELECT
        FROM pg_database
        WHERE datname = 'airflow_meta'
    ) \ gexec
