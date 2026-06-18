# Ubuntu Python 3.12 CI Constraints

This file is a placeholder for an exact dependency constraints capture from the
validated Ubuntu Python 3.12 CI environment.

Do not convert this file into a pinned `.txt` constraints file by guessing
versions. Generate it from a validated environment:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -m pip freeze > constraints/py312-ubuntu-ci.txt
python setup.py build_ext --inplace
python -m pytest -q
```

After validation, commit the generated `constraints/py312-ubuntu-ci.txt`, update
`constraints/README.md`, `docs/FILE_SCOPE.md`, and `docs/WORKLOG.md`, and remove
this placeholder if it is no longer needed.
