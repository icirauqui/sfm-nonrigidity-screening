# Development and workspace integration

This repository owns the screening algorithms and reproduction tools. The
MeshMeld research workspace consumes it as a Git submodule at
`third_party/sfm-nonrigidity-screening`; it must not copy the kernels into its
application or dated experiment directories. There is no native C++/Qt integration.

## Independent checkout

The source repository is public and can be cloned without authentication:

```sh
git clone https://github.com/icirauqui/sfm-nonrigidity-screening.git
cd sfm-nonrigidity-screening
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python tools/check_source.py
python tools/check_layout.py
python -m unittest discover -s tests -v
python tools/check_reference.py --output outputs/reference.json
```

The source check verifies the maintained file manifest before execution; the
reference check separately verifies closed numerical inputs and decisions.
Tests are arithmetic/solver checks, not new scientific validation. Strict scalar
and model-score replay have the backend limits in [numerics.md](numerics.md).
Never hide a reference failure by changing tolerances, skipping a failing Linux
test, or replacing the closed expected outputs with a fresh run.

## Workspace consumer

From the MeshMeld workspace root:

```sh
git submodule update --init --recursive third_party/sfm-nonrigidity-screening
python3.12 -m pip install -e third_party/sfm-nonrigidity-screening
python3.12 experiments/nonrigidity_screening.py --help
```

`git submodule update` selects the exact workspace commit. The workspace loader
uses that source tree, or a matching installed distribution, and checks all
package module hashes against `SCREENING_DEPENDENCY.json` before importing.
The installed fallback supports consumers without a workspace clone; it does
not silently accept another algorithm version. Research inputs remain separate
from the submodule and native MeshMeld builds do not require this Python package.

## Changes and packaging

Run package tests and the compact check after code changes. Numerical kernels
are tied to the article's frozen contracts: preserve the original methods and
adverse outcomes; name materially different successors separately. Requalify
algorithm changes before updating the workspace module hashes and commit lock.
The source manifest is `provenance/source_files.json`; after an intentional
change, review and refresh affected hashes, then run `tools/check_source.py`.
This manifest is not the historical kernel-equivalence or numerical-replay record.

Build distributions in the virtual environment:

```sh
python -m pip install build
python -m build
python -m pip install --force-reinstall --no-deps dist/*.whl
python -m unittest discover -s tests -v
python tools/check_reference.py --output outputs/installed_reference.json
```

CI builds the wheel/sdist, installs the wheel, and runs these source, unit and
compact-reference checks on Linux with single-thread BLAS settings. A successful
CI run does not establish full model-to-score or artwork reproduction.

Commit/push standalone changes first. In the workspace, record the new submodule
commit and update its dependency lock, run the integration checks, then commit
and push the workspace. Ordinary updates must not use `--remote` to bypass the
recorded scientific revision. Keep input ZIPs, virtual environments, build outputs
and caches out of Git. Do not upload the private historic capsule as a substitute
for the selected, separately attributed reproduction inputs.

Public source access is verified. Input-bundle deposition, a DOI and journal
submission are separate publication steps; a GitHub source push does not perform
them. See [architecture.md](architecture.md) for canonical module locations and
backward-compatible imports.
