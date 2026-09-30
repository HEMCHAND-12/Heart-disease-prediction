PYTHON ?= .venv/bin/python

.PHONY: reproduce reproduce-priority robustness report test

reproduce:
	mkdir -p results
	nohup $(PYTHON) scripts/reproduce.py --force-splits --phase1-priority --split both --include-tuned >> results/phase1.log 2>&1 & echo $$! > results/phase1.pid

reproduce-priority:
	mkdir -p results
	nohup $(PYTHON) scripts/reproduce.py --force-splits --phase1-priority --split paper_faithful --include-tuned >> results/phase1.log 2>&1 & echo $$! > results/phase1.pid
	$(PYTHON) scripts/report_results.py

robustness:
	mkdir -p results
	nohup $(PYTHON) scripts/reproduce.py --phase1-priority --split predictor_grouped >> results/robustness.log 2>&1 & echo $$! > results/robustness.pid
	$(PYTHON) scripts/report_results.py

report:
	$(PYTHON) scripts/report_results.py

test:
	$(PYTHON) -m pytest -q