# Release Visibility And Citation Checklist

Use this checklist before tagging a public release. It is intentionally about
discoverability, citation hygiene, and release communication; it does not
change package behavior.

## Metadata Consistency

- Confirm the release version and date are final before editing metadata.
- Confirm `pyproject.toml` `project.version` matches the release version.
- Confirm `CITATION.cff` `version` and `date-released` match the release
  version and release date.
- Confirm `codemeta.json` matches `pyproject.toml` and `CITATION.cff` for
  version, authors, license, repository URL, citation URL, keywords, and
  description.
- Confirm `CHANGELOG.md` has user-facing changes for this release, not only
  internal maintenance notes.
- Do not add DOI, publication, funding, ORCID, benchmark, platform-support, or
  provenance claims unless the underlying public source exists and is cited in
  the repository metadata.

## Discoverability

- Confirm the top README paragraph still contains the exact search phrases
  `Frontier decoder`, `quantum LDPC`, `logical maximum-likelihood`, and
  `boundary states`.
- Confirm the README quickstart still works for a fresh user and points to
  supported console scripts.
- Confirm paper and reproducibility notes are current:
  `docs/REPRODUCIBILITY.md`, `paper/README.md`, `paper/plots/README.md`, and
  `paper/plots/manifest.csv`.
- Confirm any social preview image configured in GitHub repository settings is
  current. If no social preview image is configured, record that fact rather
  than claiming one exists.

## Validation

- Run the release smoke tests from `docs/RELEASE.md`.
- At minimum, confirm these commands pass in the release environment:

```bash
python setup.py build_ext --inplace
python -m pytest -q
frontier-smoke --K 16 --Delta 100 --shots 3
frontier-dem-info --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder
frontier-dem-info --backend bravyi_depth7 --p-location 0.001 --column-order deadline_reorder
```

## Release Notes And Archive

- Draft GitHub release notes that include:
  - the associated paper arXiv link,
  - a short quickstart,
  - what changed since the previous release,
  - any compatibility notes or known limitations.
- If Zenodo integration is enabled for the repository, reserve or confirm the
  Zenodo DOI step according to the repository owner workflow, then update
  `CITATION.cff` and related metadata only after the DOI exists.
- If Zenodo integration is not enabled, record that no Zenodo DOI is assigned
  for this release.

## Announcement Drafts

- Prepare short announcement text for X/Twitter.
- Prepare short announcement text for Mastodon or Bluesky.
- Prepare a mailing-list announcement if a relevant mailing list will be used.
- Prepare a QEC Slack or Discord announcement if an appropriate channel is
  available and announcement there is welcome.
- Keep announcement text consistent with `CITATION.cff`, `CHANGELOG.md`, and
  the GitHub release notes; do not add unverified performance, adoption, DOI,
  or publication-status claims.
