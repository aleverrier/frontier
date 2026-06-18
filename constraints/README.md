# Constraints

This directory is for known-good dependency constraints used to reproduce a
validated environment. Constraints are not package installation requirements;
`pyproject.toml` remains the flexible install specification.

The current committed constraints file is:

- `constraints/py314-macos-validated.txt`: exact dependency and build-tool
  versions from the MacBook Python 3.14.2 validation environment used for this
  public-release cleanup.

Use it with:

```bash
python -m pip install -e . -c constraints/py314-macos-validated.txt
```

For another platform, capture constraints from the active validated environment:

```bash
python -m pip freeze > constraints/<python-platform-label>.txt
```

Then validate:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e . -c constraints/<python-platform-label>.txt
python setup.py build_ext --inplace
python -m pytest -q
```
