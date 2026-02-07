setup:
	pip install -e .

dev:
	pip install -e .[dev]

serve:
	python -m efflab.engine.server_fastapi

bench:
	python -m efflab.profiler.collectors

distill:
	python -m efflab.distill.trainer

quantize:
	python -m efflab.distill.quantize.bnb --bits 4

test:
	python -m pytest -q
