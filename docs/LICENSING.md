# Licensing

This repository is released under the Apache License 2.0. See the root
`LICENSE` file for the license text.

Apache-2.0 applies to repository code and documentation unless a file states a
different license or provenance constraint. Third-party dependencies and any
explicitly marked third-party files or assets remain under their own licenses.

This license choice assumes approval from the relevant copyright holders and
institutions. This note records the repository policy; it is not legal advice.

## Why Apache-2.0?

Apache-2.0 is a standard permissive software license. It is appropriate for
software code, includes an explicit patent grant, and usually creates lower
reuse friction than copyleft or NonCommercial Creative Commons licenses.

## Third-Party Context

- `quantumgizmos/ldpc`, the upstream project for the `ldpc` dependency used by
  this repo, uses the MIT License.
- `ionq-publications/BeamSearchDecoder` uses CC BY-NC-SA 4.0 for that
  repository, while explicitly keeping vendored `ldpc` files under MIT.
- This repository uses Apache-2.0 for its software and documentation. Do not
  use CC BY-NC-SA as a software-code license for this repo, and do not relicense
  third-party material that carries its own license or provenance constraints.

When adding vendored third-party code or assets, preserve upstream notices and
record the provenance in `NOTICE` or in this document.

## SPDX Headers

New repository-owned source files should include an Apache-2.0 SPDX header, for
example:

```python
# SPDX-License-Identifier: Apache-2.0
```

Do not add repository SPDX headers to third-party files or research assets with
unclear provenance unless their license status has been confirmed.
