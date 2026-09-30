PYTHON ?= .venv/bin/python

.PHONY: reproduce reproduce-priority robustness report test

reproduce:
	$(PYTHON) scripts/reproduce.py --force-splits --split stratified --include-tuned
	$(PYTHON) scripts/reproduce.py --split grouped --priority-only
	$(PYTHON) scripts/report_results.py

reproduce-priority:
	$(PYTHON) scripts/reproduce.py --force-splits --priority-only --include-tuned
	$(PYTHON) scripts/report_results.py

robustness:
	$(PYTHON) scripts/reproduce.py --split grouped --priority-only
	$(PYTHON) scripts/report_results.py

report:
	$(PYTHON) scripts/report_results.py

test:
	$(PYTHON) -m pytest -q