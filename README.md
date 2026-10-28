# SfM Nonrigidity Screening

Reference implementations and reproducible experiments for **nonrigidity screening
from global and local structure-from-motion evidence**.

The package provides three alternative diagnostics:

- **Global modal residuals:** screen one existing estimated SfM model after removing
  point, camera and affine-temporal nuisance responses. Modal energy is the primary
  readout; the optional studentized statistic and elementary controls are retained.
- **RDSC:** robust affine detrending and spatial coherence of aligned global/local
  point discrepancies, compared with a deterministic shuffle reference.
- **BASC-C:** calibrated expert composition using the declared provenance of local
  cameras, with an RDSC fallback and explicit abstention.

These are Python research implementations. They consume existing reconstructions;
reconstruction disagreement does not uniquely identify physical deformation.
BASC-C composes local experts and does not include the global modal diagnostic.
See [methods and limitations](docs/methods.md), [input contracts](docs/inputs.md)
and the [source architecture](docs/architecture.md). Main API/CLI entry points
stay at the package root; methods, local components, shared code and comparators
have separate subpackages.

## Install

Use Python 3.12 and a virtual environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

NumPy 2.2.6 and SciPy 1.16.1 are pinned. The numerical reference used Python
3.12.13 on Linux. Structural checks also run on Python 3.12.14/macOS, but strict
retained scalar replay requires the qualified Linux numerical backend.
For plotting install `python -m pip install '.[figures]'`. The optional native
M3C2 wrapper requires `python -m pip install '.[m3c2]'` and its separate runtime
qualification. The Dockerfile is a build recipe; it does not claim to recreate
an already archived image digest. See [reproduction](docs/reproduction.md).

## Small working example

The following creates an arithmetic fixture, not a research dataset:

```sh
python examples/create_fixture.py --output outputs/example
sfm-nonrigidity rdsc --pairs outputs/example/pairs.npz --output outputs/rdsc.json
sfm-nonrigidity global --model outputs/example/model --requested-views 6 --output outputs/global.json
sfm-nonrigidity compose --input outputs/example/compose.json --output outputs/composition.json
```

For a supplied COLMAP text model:

```sh
sfm-nonrigidity global --model /path/to/text_model --requested-views 90 --output outputs/model_scores.json
```

A score is not a calibrated detection. Global cutoffs must be supplied from the
appropriate static controls; the example does not estimate general sensitivity.
Outputs are created without overwriting an existing result.

For a self-checking run of all six commands, use
`python examples/run_example.py --output outputs/complete_example`.
The [usage guide](docs/usage.md) documents input/output fields, Python calls,
Docker use, errors and scientific refusals.

## Local preparation and composition

`sfm-nonrigidity refine` performs deterministic point-only refinement with fixed
poses, intrinsics and tracks. `local` computes all local components from prepared
model pairs. `sequence` aggregates requested ordinal windows with their missing
values and supplied calibration membership. `compose` operates on already
calibrated expert intervals. Run `sfm-nonrigidity COMMAND --help` for arguments.

The primary comparison retains the **original-export matched-support extent**
when scoring fitted points. This is explicit in the full replay tool. For direct
aligned arrays use `rdsc --normalization-extent VALUE` to supply that extent;
otherwise normalization uses the supplied reference positions. These choices
must not be mixed under the same frozen threshold.

The CLI defaults to the exact `epc` scalar profile. Its shuffled reference
preserves the evaluated `x + residual` arithmetic path. `--profile original`
selects the unchanged residual-only RDSC primitive. In Python these are explicit
as `epc_scores` and `rdsc_scores`; mathematically equivalent intermediate
expressions must not be substituted when reproducing floating-point outputs.

## Checks and reproduction

```sh
python tools/check_source.py
python tools/check_layout.py
python -m unittest discover -s tests -v
python tools/check_reference.py --output outputs/reference.json
```

The compact reference contains 31 projected global matrices, one retained global
refusal, 216 closed local-window score records and 96 calibrated sequence
comparisons. The compact check replays global scalar readout and local decisions;
full model-to-score replay is a separate command using selected input bundles.
See [reproduction instructions](docs/reproduction.md) for input preparation,
all-primary numerical replay, exact artwork generation and comparison controls.

[Numerical validation](docs/numerics.md) distinguishes source-migration equality,
full decision reproduction and strict historic score parity. Small JAPD/edge
backend differences remain reported; historic tolerances are not relaxed.

## Scope and availability

The study's controlled core has four exposed synthetic families / twelve
acquisitions. Positive appearance transfer reuses two families; it is not an
independent anatomical cohort. Coherent amplitude matches primary RDSC decisions;
RDSC-specific utility gates fail. There is no claim of uniform superiority,
physical displacement accuracy, population specificity or clinical validation.
The compact check and manufactured tests do not change these limits.

Source and selected closed numerical references are present here. The separate
[public reproduction inputs](https://huggingface.co/datasets/icirauqui/sfm-nonrigidity-screening-reproduction/tree/0621f69360f78f5379afffb872532648d1e6e85a) are dataset version 0.1.0, revision
`0621f69360f78f5379afffb872532648d1e6e85a`. They supply selected primary models/points and display-only assets;
see [download and replay instructions](docs/reproduction.md). The data deposit
has no DOI assigned here; its fixed revision identifies the published files.
The companion benchmark article is [arXiv:2610.35621v1](https://arxiv.org/abs/2610.35621v1).
The benchmark and separate reproduction resources are
[10.57967/hf/10850](https://doi.org/10.57967/hf/10850) and
[10.57967/hf/10851](https://doi.org/10.57967/hf/10851); these identify different
resources. Historic E16/J08 raw imagery is unavailable; selected model replay
is not end-to-end image reconstruction or replay of every historical experiment.

## License and citation

Software is GPL-3.0-or-later; see [LICENSE](LICENSE), [third-party notices](THIRD_PARTY.md)
and [CITATION.cff](CITATION.cff). Numerical reference records have the same license
unless a separate input license is explicitly stated. Benchmark display inputs
retain their source-data license and attribution. Upstream LOG source is not
redistributed: a hash-checked external adapter is provided separately.

The [repository/release guide](docs/release.md) lists all supplied material and
the exact public input version and validation/submission boundaries.

Developer setup, packaging checks and workspace submodule use are described in
[the development guide](docs/development.md).

The source repository and selected input deposit are public. A deposit DOI,
external review, new experiments, native MeshMeld integration and journal
acceptance are not implied.
