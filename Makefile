.PHONY: build-native test smoke dem-info lint typecheck clean

build-native:
	python setup.py build_ext --inplace

test:
	python -m pytest -q

smoke:
	python -m tools.frontier_decoder --K 16 --Delta 100 --shots 3

dem-info:
	python -m tools.dem_loader --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder

lint:
	python -m ruff check .

typecheck:
	python -m mypy frontier grosscode tools

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf build dist .pytest_cache *.egg-info
