# Build from the repo root:  docker compose -f ops/docker-compose.ops.yml build
FROM apache/airflow:2.10.5-python3.12
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
USER airflow
# The project pins newer numpy/pandas than Airflow supports, so the project code runs in its own
# virtualenv. DAGs call /home/airflow/pg-venv/bin/python; Airflow itself is untouched.
RUN python -m venv /home/airflow/pg-venv
COPY --chown=airflow:root requirements.txt /tmp/req/requirements.txt
COPY --chown=airflow:root ops/requirements-ops.txt /tmp/req/requirements-ops.txt
RUN /home/airflow/pg-venv/bin/pip install --no-cache-dir -r /tmp/req/requirements.txt -r /tmp/req/requirements-ops.txt

# Evidently needs pandas<3, so it gets its own tiny virtualenv. drift.py calls it as a subprocess.
RUN python -m venv /home/airflow/ev-venv \
    && /home/airflow/ev-venv/bin/pip install --no-cache-dir "pandas<3" evidently
# Evidently needs pandas 2.x, so it gets its own small venv (used via EVIDENTLY_PY).
COPY --chown=airflow:root ops/requirements-evidently.txt /tmp/req/requirements-evidently.txt
RUN python -m venv /home/airflow/ev-venv \
    && /home/airflow/ev-venv/bin/pip install --no-cache-dir -r /tmp/req/requirements-evidently.txt
