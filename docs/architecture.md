# Source organization

The `src/` layout retains a standard installable `sfm_nonrigidity` package. Its
root contains the main public entry points, with algorithms grouped by purpose:

```text
src/
  README.md
  sfm_nonrigidity/
    __init__.py                 public functions and version
    __main__.py                 python -m sfm_nonrigidity
    api.py                      public Python facade
    cli.py                      six-command CLI
    methods/
      global_modal/             global model response and modal projection
      rdsc/                     RDSC, shuffled controls and exact EPC profile
      basc_c/                   expert routing and temporal/calibration readout
    local/                      fitting, JAPD, edge and transfer components
    common/                     COLMAP IO, alignment, precision, linear/QR response
    comparators/                classical distances, M3C2 and external LOG adapter
    _compat.py                  aliases for earlier flat module imports
    m3c2.py                     lazy compatibility alias for optional comparator
```

New code should import the appropriate canonical subpackage or public API. For
example, `sfm_nonrigidity.methods.global_modal.global_projection` contains the
modal operator; `sfm_nonrigidity.methods.rdsc.rdsc` contains original RDSC; and
`sfm_nonrigidity.methods.basc_c.sequence` contains temporal composition. Shared
primitives are not copied into methods. The CLI and public functions retain
their names and contracts.

Earlier imports such as `sfm_nonrigidity.geometry`, `sfm_nonrigidity.rdsc` and
`sfm_nonrigidity.sequence` remain compatibility aliases. They refer to the exact
same module objects as their canonical paths. This matters because the retained
local response temporarily installs a process-local projection callback. Loading
independent copies under old/new names would change its behavior. Tests check
both module identity and callback sharing. The optional M3C2 alias loads its native
dependency only on explicit request; importing the core package does not require it.

The layout check runs under the supported Python 3.12 AST schema.
`provenance/package_layout.json` maps all 24 former module names to canonical
paths. `tools/check_layout.py` checks all 70 numerical function/class bodies in
23 modules against the original migration AST records. Only module paths and
top-level imports changed; numerical definitions, constants, thresholds, seeds,
refusals and reference files retain their values. This organization change is
software version 0.1.1. The separate input bundle names remain v0.1.0 because
their bytes and contracts are unchanged.

Top-level `tools/`, `examples/`, `tests/`, `docs/`, `provenance/` and
`reproduction/reference/` retain their distinct roles. Package distributions
include the subpackages and reproduction/development material; the native
MeshMeld application consumes no copied algorithm code.
