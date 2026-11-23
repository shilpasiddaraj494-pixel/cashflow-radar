SOURCE ?= synthetic
export PYTHONPATH := src
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: install data train train-ibm score api web test lint mlflow up down clean

install:            ## Python + frontend dependencies
	pip install -e ".[dev]"
	cd frontend && npm ci

data:               ## Generate the synthetic invoice book
	python -m cashflow_radar.data.synthetic

train:              ## Train + evaluate + log to MLflow (SOURCE=synthetic|ibm)
	python -m cashflow_radar.train --source $(SOURCE)

train-ibm:
	$(MAKE) train SOURCE=ibm

score:              ## Score open invoices and load the worklist into the database
	python -m cashflow_radar.score --source $(SOURCE)

api:                ## Run the API on :8000
	uvicorn cashflow_radar.api.main:app --reload --port 8000

web:                ## Run the dashboard on :5173 (proxies /api to :8000)
	cd frontend && npm run dev

test:               ## Python + frontend tests
	python -m pytest -q
	cd frontend && npm test

lint:
	ruff check src tests
	cd frontend && npm run typecheck

mlflow:             ## Browse experiment runs on :5000
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

up:                 ## Full stack in Docker: Postgres + trainer + API + dashboard on :8080
	docker compose up --build

down:
	docker compose down -v

clean:
	rm -rf artifacts mlruns data/synthetic
