# Constraints

This directory is for known-good dependency constraints used to reproduce a
validated environment. Constraints are not package installation requirements;
`pyproject.toml` remains the flexible install specification.

Install with a validated constraints file using:

```bash
python -m pip install -e . -c constraints/<validated-environment>.txt
```

The current committed exact constraints file is:

- `constraints/py314-macos-validated.txt`: exact dependency and build-tool
  versions from the MacBook Python 3.14.2 validation environment used for this
  public-release cleanup.

The `py314-macos-validated.txt` file is one available validated environment,
not a default for all users.

The Ubuntu Python 3.12 CI environment is validated by GitHub Actions, but this
checkout does not yet include a truthful exact pin capture from that runner.
`constraints/py312-ubuntu-ci.TODO.md` records the command sequence to generate
and validate that file without guessing pins.

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
