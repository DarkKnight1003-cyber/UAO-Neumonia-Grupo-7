.PHONY: install run test docker-build docker-run clean

install:
	uv sync --frozen

run:
	uv run python -m src.detector_neumonia

test:
	uv run python -m pytest -q test

docker-build:
	docker build -t uao-neumonia-grupo-7 .

docker-run:
	docker run --rm -e DISPLAY=host.docker.internal:0.0 -v "${PWD}\conv_MLP_84.h5:/app/conv_MLP_84.h5:ro" uao-neumonia-grupo-7

clean:
	rm -rf .venv __pycache__ .pytest_cache historial.csv Reporte*.jpg Reporte*.pdf