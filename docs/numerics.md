# Numerical equivalence and backend limits

The migration preserves 70 numerical function/class definitions across 23
modules. Their ASTs match the original implementations, allowing only one
relative import and explicit dependency aliases in the EPC scalar adapter.
`provenance/kernel_equivalence.json` records those identities. On the sensitive
retained static window, direct original-versus-migrated execution on the same
Linux ARM64 backend gives **zero score difference** for all six local components
in both independent and inherited camera regimes. No operator was altered to
make a result favorable.

Fifteen numerical tests pass on Linux, including independent point optimization,
finite-difference gradients, dense-versus-streamed projection, transfer bounds,+composition and the retained EPC scalar fixture. Structural tests also run on
macOS; the strict retained scalar fixture is explicitly Linux-specific.

The complete model replay uses 32 globals and 216 local-window/regime records.
On the tested Linux ARM64 backend, all global/scalar scores satisfy the retained
tolerances, and all calibrated global and local decisions match. However, **92
JAPD/edge score entries exceed the historic 1e−8/1e−10 local tolerance**. The
largest absolute score difference is 4.674566005391512e−5. The strict check is
therefore a numerical-reference mismatch, not a pass. An x86_64 container with
Python3.12.13 also exhibits small geometry-component differences on this host.
Changing architecture or Python version alone does not guarantee the original
BLAS/CPU floating-point path. macOS can additionally change shuffled scores;
the same differences occur with the unmodified original code.

The original controls, tolerances, cutoffs and stored reference scores remain
unchanged. The replay tool offers two explicit comparison modes:

- `--comparison strict` (default) preserves the original score tolerances and
  returns failure after saving discrepancies when they are exceeded.
- `--comparison decisions` checks all decisions and still writes the full
  numerical mismatch report. Its status is
  `DECISIONS_MATCH_NUMERICAL_REFERENCE_MISMATCH` when applicable; it is not
  relabeled as strict score parity.

The tested backend preserves all 96 local sequence/calibration definitions and
the 480 core global decision records plus 100 transfer decisions. Core global
records duplicate two local-regime labels: 240 distinct core checks / 480
records, plus 100 transfer checks. This remains an exposed, dependent evaluation
population, not new independent scientific confirmation.

These observations qualify software relocation and distinguish numerical from
decision reproduction. They do not establish numerical invariance, robustness
to new imaging conditions, physical validity or detector superiority. Use the
historic qualified runtime/hardware for its strict score contract, and requalify
another backend before transferring thresholds to new inputs. The current
dependency pins, source identities, exact figure replay and retained discrepancies
are provided so this boundary can be inspected rather than hidden.
