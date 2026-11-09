# PromptGuard

Prompt injection detection API: classifies a prompt as benign or injection, with MLOps pipeline (DVC, MLflow, Docker, CI/CD, Airflow, Evidently) on AWS.

## Setup
```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
dvc pull
```

## Run
```bash
python app/train.py
python app/eval.py --promote
python app/inference.py predict "Ignore previous instructions" --model champion
```

Status: data, training, evaluation and inference done. API, Docker, CI/CD, Airflow, monitoring in progress. See `Project Proposal.pdf`.
