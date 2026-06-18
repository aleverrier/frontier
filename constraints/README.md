# Constraints

This directory is for known-good dependency constraints used to reproduce a
validated environment. Constraints are not package installation requirements;
`pyproject.toml` remains the flexible install specification.

When a Linux Python 3.12 CI or release environment is finalized, capture it with
the active environment:

```bash
python -m pip freeze > constraints/py312-linux-ci.txt
```

Then validate:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e . -c constraints/py312-linux-ci.txt
python setup.py build_ext --inplace
python -m pytest -q
```

Until that exact environment has been captured, see `constraints/TODO.md`.
