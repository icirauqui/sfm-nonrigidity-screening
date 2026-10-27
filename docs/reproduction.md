# Reproduction

## Compact numerical check

Install the package, then run:

```sh
python tools/check_source.py
python tools/check_layout.py
python -m unittest discover -s tests -v
python tools/check_reference.py --output outputs/reference.json
```

This verifies every small reference-file hash, recomputes scalar statistics from
31 projected 61×61 matrices (one model had a retained rank refusal), and exactly
replays all 96 sequence/calibration definitions from 216 closed local scores.
Those definitions include deletion folds and dependent camera-regime records;
they are not 96 independent scenes. This stage does not recompute matrices from
text models or reconstruct images.

## Full primary model-to-score replay

Large text-model/point inputs are separate from Git. The author-prepared bundle
has `INPUT_MANIFEST.json` and `CONTRACTS.json`, with neutral relative paths and
hashes for all selected inputs. A public asset URL/DOI remains to be issued.
From the author's preserved review capsule, the bundle is prepared by:

```sh
python tools/prepare_reproduction_inputs.py --capsule /path/to/extracted_capsule --output /new/primary_inputs
```

Only explicit primary models/point arrays are copied: 32 requested global models,
108 local windows, independent/inherited fitted points, no RGB or evaluator truth.
Once the bundle is obtained or prepared:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tools/replay_models.py --inputs /path/to/primary_inputs --mode all --workers 4 --comparison decisions --output outputs/full_replay
```

Workers load only their explicit model/point allowlist and run the standalone
installed package. Expected outputs are read in the parent only after inference
completes. The replay compares all available scores and global ranks/df, preserves
whole-model refusals, and checks every requested BASC-C/component/global decision.
The original strict tolerances remain rtol1e−7/atol1e−8 for global scalar scores
and rtol1e−8/atol1e−10 for local scores. Another backend can exceed them for JAPD
and edge scores while preserving decisions. The explicit decision mode writes
`DECISIONS_MATCH_NUMERICAL_REFERENCE_MISMATCH` and saves every discrepancy; it
does not turn those differences into a strict numerical pass. Use
`--comparison strict` for the original score contract (the default).
See [numerical validation](numerics.md) for the retained failures and direct
original-versus-migrated equivalence test.
Use separate processes; the retained local response temporarily sets a process-local
projection callback and is not intended for concurrent threads.

The primary replay supplies the frozen original-export normalization extent for
fitted local residuals. Preparation and all negative/missing outcomes remain
unchanged. The original-export ablation, older historical cohorts and manufactured
utility campaigns are distinct evidence and retain their own frozen capsule
contracts; this command does not claim to rerun all historical experiments.
Raw E16/J08 payloads were deleted, so full image-to-result replay is unavailable.

The qualified historical Linux runtime was Python3.12.13 / NumPy2.2.6 /
SciPy1.16.1, image ID
`sha256:88e6523c511d52dc76b09d9d79946ab377278af56061d2126727028a88ade9bd`.
That image ID is provenance, not a public container download. The Dockerfile
provides a new build recipe; a new image needs its own qualification. Timings
from another host/environment are not replacements for published timing records.
Strict model-to-score reference reproduction requires the qualified Linux
numerical backend. The same original and migrated kernels can produce different
shuffle-based scores on macOS ARM64, despite identical source mathematics,
NumPy/SciPy versions and input bytes. Migration checks retain this adverse result;
they do not widen tolerances or replace the original reference scores. The
retained scalar fixture is therefore a Linux-specific test. Other structural,
solver and compact decision checks also run on macOS. Some macOS matrix operations
emit floating-status warnings with finite outputs; logs remain inspectable.
Use the qualified container for strict replay and requalify any other backend.

## Figures and tables

Install plotting extras and obtain the separate display-only input bundle. It
contains only the selected geometry/RGB, closed model-display arrays and closed
result records needed for the nine figures/graphical abstract. These are not
inference inputs. All files are hash-bound:

```sh
MPLCONFIGDIR=outputs/mpl-cache XDG_CACHE_HOME=outputs/cache \
python tools/render_artwork.py --inputs /path/to/artwork_inputs --output outputs/artwork
```

The renderer regenerates complete PDF/PNG figures with the retained common-scale
atlas, enlarged annotations and explicit near-threshold decision signs. The
associated caption file is `docs/figure_captions.md`. Production crops require the
complete figure captions. Result-table generation and closed comparison controls
are separate from model inference; see `tools/summarize_results.py`.

## Optional comparators

The native M3C2 adapter is `sfm_nonrigidity.comparators.m3c2.compute(x,y)`
(the previous flat import remains compatible), requiring py4dgeo1.2.0
in its qualified environment. Primary py4dgeo runs were macOS ARM64 and are not
assumed installed in the Linux numerical image. All explicit support/admission
rules and same-support raw controls remain in the adapter. LOG implementation
redistribution is restricted as described in `THIRD_PARTY.md`; its retained closed
results can be inspected without treating them as a new full-source replay.
