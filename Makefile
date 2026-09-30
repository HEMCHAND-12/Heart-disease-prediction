PYTHON ?= .venv/bin/python

.PHONY: reproduce reproduce-priority robustness report test

reproduce:
	mkdir -p results
	nohup sh -c '$(PYTHON) scripts/reproduce.py --force-splits --phase1-priority --split both --include-tuned && $(PYTHON) scripts/report_results.py' >> results/phase1.log 2>&1 & echo $$! > results/phase1.pid

reproduce-priority:
	mkdir -p results
	nohup sh -c '$(PYTHON) scripts/reproduce.py --force-splits --phase1-priority --split paper_faithful --include-tuned && $(PYTHON) scripts/report_results.py' >> results/phase1.log 2>&1 & echo $$! > results/phase1.pid

robustness:
	mkdir -p results
	nohup sh -c '$(PYTHON) scripts/reproduce.py --phase1-priority --split predictor_grouped && $(PYTHON) scripts/report_results.py' >> results/robustness.log 2>&1 & echo $$! > results/robustness.pid

report:
	$(PYTHON) scripts/report_results.py

test:
	$(PYTHON) -m pytest -q