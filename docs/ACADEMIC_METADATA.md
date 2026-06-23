# Academic Metadata

This file records the public academic metadata currently declared by this
repository. It avoids inferred software DOI, ORCID, funding, and asset-origin
claims.

## Citation Metadata

- Software title: `Frontier decoder for quantum LDPC codes`.
- Software description: Frontier decoder for quantum LDPC codes and detector
  error models.
- Software authors declared in `CITATION.cff`: Anthony Leverrier and Rüdiger
  Urbanke.
- Software version: `0.1.0`.
- Release date: `2026-06-18`.
- Repository: `https://github.com/aleverrier/frontier`.
- License: Apache-2.0.
- CodeMeta: `codemeta.json`.
- ORCIDs: none declared.
- Institution or lab: none declared.
- Software DOI: none assigned in this repository.
- Related paper/preprint: Anthony Leverrier and Rüdiger Urbanke,
  "Approximating optimal decoding of quantum LDPC codes with narrow frontiers,"
  arXiv:2606.20513 [quant-ph], submitted 2026-06-18.
- Related paper URL: `https://arxiv.org/abs/2606.20513`.
- Related paper arXiv DOI: `10.48550/arXiv.2606.20513`.
- Citation policy: cite the software using `CITATION.cff`; cite the paper
  separately using `arXiv:2606.20513`.
- Search keywords:
  - quantum LDPC
  - QLDPC
  - quantum error correction
  - stabilizer codes
  - detector error model
  - logical maximum likelihood
  - coset decoding
  - dynamic programming
  - boundary state decoder
  - frontier decoder
  - bivariate bicycle code
  - LDPC codes
  - BB code
  - Gross code
  - Stim

## Funding Metadata

- Funding programme: Plan France 2030.
- Project/grant identifier: ANR-22-PETQ-0006.
- Acknowledgement text: Anthony Leverrier acknowledges the Plan France 2030
  through the project ANR-22-PETQ-0006.
- Institutional acknowledgement: Rüdiger Urbanke gratefully acknowledges the
  hospitality of the COSMIQ group at Inria, where this work was carried out
  during his sabbatical.
- LLM acknowledgement: we acknowledge the use of large language models, in
  particular OpenAI Codex, to assist with the programming, documentation, and
  testing of the frontier decoder implementation.

## Asset Provenance Status

Bundled Gross/BB144 CSS matrices, Stim circuits, and prebuilt DEM snapshots are
documented in `docs/ASSET_PROVENANCE.md` and checksummed in
`docs/ASSET_MANIFEST.md`. The repository records them as bundled research
assets for the documented workflows; no separate upstream DOI or standalone
asset license statement is declared in this repository.

## Release Metadata Policy

For citable releases, keep `CITATION.cff`, `CHANGELOG.md`,
`docs/ASSET_MANIFEST.md`, and the relevant constraints file synchronized with
the tagged source state. Do not add software DOI, ORCID, funding, or
asset-license claims unless those facts are available.

## Metadata Authority Order

- Package version source: `pyproject.toml`.
- Citation metadata source: `CITATION.cff`.
- Release notes: `CHANGELOG.md`.
- Internal maintenance log: `docs/WORKLOG.md`.
- Asset checksums: `docs/ASSET_MANIFEST.md`.
- Paper-plot data manifest: `paper/plots/data/MANIFEST.md`.
- Paper figure manifest: `paper/plots/manifest.csv`.
- CodeMeta export: `codemeta.json`, derived from `pyproject.toml`,
  `CITATION.cff`, `ACKNOWLEDGEMENTS.md`, and repository docs.

Maintainer metadata is not declared in `pyproject.toml` because this repository
does not currently declare a separate software maintainer or maintainer contact.
