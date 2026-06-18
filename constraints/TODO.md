# Constraints TODO

No exact `constraints/py312-linux-ci.txt` file is committed yet because the
current exact Linux Python 3.12 release environment has not been truthfully
captured in this pass.

Before an archival release:

1. Create a clean Python 3.12 Linux environment.
2. Install and validate the project.
3. Run `python -m pip freeze > constraints/py312-linux-ci.txt`.
4. Re-run the validation commands from `docs/REPRODUCIBILITY.md` with
   `python -m pip install -e . -c constraints/py312-linux-ci.txt`.
