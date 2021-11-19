format:
	poetry run black spectral_metric

lint:
	poetry run flake8 spectral_metric

mypy:
	poetry run mypy spectral_metric

test: lint mypy
	poetry run pytest tests
