PY ?= python

.PHONY: run build test clean

run:
	$(PY) run.py

build:
	$(PY) run.py --no-serve

test:
	$(PY) -m pytest tests -q

clean:
	rm -rf data/kept.db data/models data/metrics.json
