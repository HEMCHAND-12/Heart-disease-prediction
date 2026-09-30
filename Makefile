PYTHON ?= .venv/bin/python

.PHONY: all baseline experiments report test

all: baseline experiments

baseline:
	$(PYTHON) -m scripts.baseline_parsel

experiments:
	mkdir -p results
	nohup sh -c '$(PYTHON) -m scripts.run_experiments --phase1-priority --split both --include-tuned && $(PYTHON) -m scripts.report_results' >> results/phase1.log 2>&1 & echo $$! > results/phase1.pid

report:
	$(PYTHON) -m scripts.report_results

test:
	$(PYTHON) -m pytest -q