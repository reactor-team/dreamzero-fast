# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
.PHONY: test lint

test:
	uv run --python 3.12 --with reactor-runtime==3.11.2 --with reactor-sdk==1.13.0 --with pytest --with pyyaml --with opencv-python-headless pytest

lint:
	uv run --with ruff ruff check dreamzero.py dreamzero_model.py model_types.py benchmark.py tests ../client
	uv run --with ruff ruff format --check dreamzero.py dreamzero_model.py model_types.py benchmark.py tests ../client
