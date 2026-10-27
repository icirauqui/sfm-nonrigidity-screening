# Using the reference implementation

Install from the repository root as described in [the README](../README.md).
Commands below assume an activated Python 3.12 environment. No workspace,
MeshMeld application, GPU, internet connection or research inputs are needed
once the base package dependencies have been installed.

## Complete arithmetic example

```sh
python examples/run_example.py --output outputs/complete_example
```

This runs all six CLI commands and checks their outputs, saving
`EXAMPLE_CHECK.json`. It makes a six-view, 120-point pinhole fixture, scores aligned
arrays, refines points and compares the fitted model with itself. Identical-camera
transfer gives `[0.5, 0.5]`. Composition and temporal aggregation have separate
hand-set scores: the temporal example is not a sequence inferred from the single
fixture model. None of these examples estimates detector performance.

For individual commands, first create the fixture:

```sh
python examples/create_fixture.py --output outputs/example
sfm-nonrigidity global --model outputs/example/model --requested-views 6 --output outputs/global.json
sfm-nonrigidity rdsc --pairs outputs/example/pairs.npz --output outputs/rdsc.json
sfm-nonrigidity refine --model outputs/example/model --output outputs/refined.npz
sfm-nonrigidity local --global-model outputs/example/model --local-model outputs/example/model --global-points outputs/refined.npz --local-points outputs/refined.npz --provenance inherit_global --window-index 0 --output outputs/local.json
sfm-nonrigidity compose --input outputs/example/compose.json --output outputs/composition.json
sfm-nonrigidity sequence --input outputs/example/sequence.json --output outputs/sequence.json
```

Each output path must be new. Successful command execution returns exit code 0;
CLI input/file errors return 2. Scientific refusal can still return 0 with an
explicit abstention or unavailable component: inspect the output, not just exit
status. `--help` describes every command's options. The same commands are available
through `python -m sfm_nonrigidity`.

## Inputs and outputs

| Command | Input | Output and interpretation |
| --- | --- | --- |
| `global` | COLMAP text triplet, optional fitted-point NPZ, requested view count | JSON `status`, support and scores; `ABSTAIN` supplies a reason. `mode_gain` is pixel-squared energy; `pixel_median`, `pixel_q95`, `null_rms` are pixels; `raw_pixel_F_stream_r0` is dimensionless. No calibrated decision is returned. |
| `rdsc` | NPZ `positions`, `residuals`: finite aligned N×3 arrays | JSON scalar scores; primary `rdsc`, coherent amplitude and raw discrepancies are normalized by the declared spatial extent. `--profile epc` is the default, with the evaluated arithmetic path. |
| `refine` | Model; optional reference supplying inherited cameras | NPZ fitted `points`, Boolean `converged` and solver diagnostics. Inspect convergence; retaining a file does not establish that every fit succeeded. |
| `local` | Global/local models, optional fitted-point NPZs, camera provenance, ordinal window index | JSON pair count, admission, component `scores` and `reasons`. No sequence-level decision. Inherited point overrides require the appropriate fixed global cameras. |
| `compose` | Explicit provenance and calibrated score/cutoff intervals | JSON `decision`, `selected_expert`, component decisions and reason. `true`/`false`/`null` mean positive/negative/abstention. |
| `sequence` | Requested ordinal window rows and explicit calibration membership | JSON `records`, with endpoints, evidence, component decisions, `basc_c`, ablations and selected expert. Missing window scores must remain missing/null. |

The exact file, coordinate, normalization, preparation, coverage and temporal
contracts are in [inputs.md](inputs.md). The methods and limitations are in
[methods.md](methods.md). CLI `local` retains original-export matched-support
normalization when fitted points are supplied. Python API users must pass that
normalization explicitly when reproducing the primary fitted-point comparison.

The primary global contract requests 90 views and admits at least 72; the
six-view example changes that declared request explicitly. A new resolution,
observation support, noise scale or scalar profile needs its own calibration.
A global score is compared with a cutoff from the declared static controls;
it is not a p-value or physical-motion measurement. BASC-C composes local
experts, not the separately reported global diagnostic.

## Python entry points

```python
from sfm_nonrigidity import load_model, global_scores, compose

model = load_model("outputs/example/model")
result = global_scores(model, requested_views=6)
if result["status"] == "SCORED":
    energy = result["scores"]["mode_gain"]  # requires separate calibrated cutoff

answer = compose("independent", {
    "rdsc": {"score": [2.0, 2.0], "cutoff": [1.0, 1.0]}
})
assert answer["decision"] is True
```

Public functions are exported from `sfm_nonrigidity`: `load_model`,
`global_scores`, `rdsc_scores`, `epc_scores`, `refine_points`, `local_scores`,
`compose`. Temporal aggregation is `sfm_nonrigidity.sequence.infer`.
Local geometry uses a process-local projection callback; parallelize separate
processes rather than concurrent threads. Do not change the cached model tuple
in place; a fresh input path/process avoids reusing stale model-reader cache.

## Closed paper results

```sh
python tools/check_reference.py --output outputs/reference.json
python tools/summarize_results.py --output outputs/comparison
```

The first replays projected scalar readouts and closed local decisions. The
second writes `comparison.json` and `comparison.csv` from existing predictions
and evaluator labels; it does not run detector inference. Full primary input
replay and artwork require separate bundles and commands in
[reproduction.md](reproduction.md). Neither command reconstructs images or
restores deleted historical raw data.

## Docker CLI

```sh
docker build -t sfm-nonrigidity:local .
docker run --rm sfm-nonrigidity:local --help
```

The image entry point is the CLI, so append `global`, `rdsc` or another command,
not `python`. Mount only the required inputs read-only and a separate writable
output directory. The Dockerfile installs the core package; it does not include
research bundles, plotting extras or the optional native M3C2 comparator. It is
a new runtime recipe, not the historical qualified image. Requalify numerical
behavior before treating another image as a strict reproduction environment.

## Troubleshooting

- Existing output: use a fresh path. Commands deliberately refuse replacement.
- Missing model or schema: supply all three COLMAP text files and the documented
  observation header; do not silently convert identities or track ordering.
- Insufficient coverage/rank or failed fitting: preserve the refusal and its
  denominator. Do not discard requested cases to improve a detection rate.
- Nonfinite arithmetic or unsupported camera: inspect refusal reasons and the
  supported contracts; a new camera model needs separate implementation validation.
- Strict replay mismatch on another backend: retain the discrepancy report and
  consult [numerics.md](numerics.md). Decision parity and strict score parity are
  different checks; do not widen tolerances or replace expected values.
- Missing plotting/M3C2 modules: install the matching optional extra and qualify
  its runtime. Closed comparator results do not imply an installed comparator.
- Missing full inputs: the Git checkout includes compact references; see
  [the asset index](../release_assets/README.md). Public bundle URLs are pending.
