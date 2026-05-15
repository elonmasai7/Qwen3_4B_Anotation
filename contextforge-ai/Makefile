PYTHON ?= python
VENV ?= .venv
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest

.PHONY: setup install test annotate annotate-async worker evaluate ablation submit api dashboard all clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt

install: setup

test:
	PYTHONPATH=. $(PYTEST) tests -q

annotate:
	PYTHONPATH=. $(PYTHON) scripts/run_annotation.py --input data/raw/test.jsonl --output data/predictions/preds.jsonl

worker:
	PYTHONPATH=. $(PYTHON) scripts/run_worker.py

annotate-async:
	PYTHONPATH=. $(PYTHON) scripts/run_annotation_async.py --input data/raw/test.jsonl --output data/predictions/preds.jsonl

evaluate:
	PYTHONPATH=. $(PYTHON) scripts/run_evaluation.py --pred data/predictions/preds.jsonl --gold data/raw/gold.jsonl

ablation:
	PYTHONPATH=. $(PYTHON) scripts/run_ablation.py --config experiments/configs/ablation.yaml

submit:
	PYTHONPATH=. $(PYTHON) scripts/build_submission.py

api:
	PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	PYTHONPATH=. streamlit run dashboard/streamlit_app.py

all: test annotate evaluate submit

clean:
	rm -rf .pytest_cache __pycache__
