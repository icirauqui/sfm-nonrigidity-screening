# Input contracts

## Models

A model directory contains `cameras.txt`, `images.txt`, `points3D.txt` in COLMAP
text form. Poses are world-to-camera rotations and world-coordinate centers.
The reader recognizes the standard three-column image-observation schema and
its retained four-column extension from the schema header. Image IDs and names,
keypoint ordinals, point file order and complete tracks are significant.
Do not concatenate independent acquisitions or silently renumber tracks.

The six-element model tuple is `(poses, observations, camera_ids, cameras,
points, tracks)`. Camera entries are `(model_name, intrinsics)`. Primary global
inference supports signed PINHOLE observations with fixed intrinsics; the local
response also retains its explicitly qualified fisheye projection routines.
Unsupported or rank-deficient global models abstain as a whole. Default global
coverage is at least 72 of 90 requested views. Changing that contract requires
explicit documentation and does not inherit the study's calibration.

Optional point NPZ files contain `points` in the intended model frame and a
Boolean `converged` vector. They preserve original point order and track support.
Use `allow_pickle=False`. Inherited local point arrays use the global frame;
the CLI applies global camera poses only when that preparation is explicitly
requested with inherited provenance and a local point override. Inherited
image identities and intrinsics must match. The API `local_scores` instead
expects the fully prepared model tuple, including the appropriate poses.

## Aligned RDSC arrays

`rdsc` reads NPZ arrays `positions` and `residuals`, both finite N×3. Residuals
are aligned floating points minus reference points. Camera-first proper Sim(3)
alignment and exact track intersection are available in `sfm_nonrigidity.common.alignment` (the earlier flat import remains an alias).
The primary seed is `20260904 + window_index`, eight requested neighbors and
20 shuffles. Nearest-neighbor ties follow the retained reference implementation;
TC-RDSC was a separate numerical profile and is not silently substituted.

Normalization is q90 distance to the reference centroid. For the primary EPC
comparison, the extent is measured on the original, unrefined matched reference
support and retained after point fitting. Pass this scalar explicitly to
`epc_scores(..., normalization_extent=...)` or the CLI. Local-model admission
and component preparation remain the declared frozen contracts. A user-supplied
new support, normalization or resolution needs its own threshold qualification.

The exact primary scalar profile also preserves the original shuffled-reference
control's reconstruction of `x + residual` before subtraction. Its a²/b readout
and additive controls are in `epc_scalar`. The residual-only primitive in `rdsc`
is preserved separately. Small roundoff changes before robust iteration can
change outputs; profile choice is explicit rather than silently interchangeable.
The CLI `rdsc` and `local` default to `--profile epc`; `--profile original`
selects the residual-only route. Python `local_scores` also defaults to EPC.

## Temporal sequence input

`sequence` accepts a JSON object with exactly `rows` and `calibration`.
Each row has `index`, `sequence`, `regime` and `scores`; unavailable expert
scores are missing or null. The checked reference files provide complete examples.
A calibration definition has `id`, `sequence`, `regime`, and the explicit list
`calibration` of other sequences. Inference does not infer static controls from
names or load evaluator labels. Every calibration member must be requested.

Primary EPC: nine 15-image windows at selected-image starts
0,10,20,30,40,50,60,70,75. `index` is the ordinal 0–8, not the raw start frame.
At least eight of nine windows must be available. Transfer requires three
consecutive ordinal anchors; all other experts require two. The endpoint is
maximum over contiguous run minima, separately for interval bounds. Missing
anchors never become adjacent. BASC-C routes the resulting sequence decisions.

The primary sequence adapter accepts one frontend arm per index and rejects
repeat indices. Historical multi-arm pooling had a separate contract (maximum
bounds across arms at each requested anchor), retained in the private historical
capsule; it is not silently applied to primary EPC.

## Calibrated composition

`compose` accepts exactly `provenance` and `evidence`. Provenance is `independent`
or `inherit_global`. Each expert provides `score: [lower, upper]` and
`cutoff: [lower, upper]`, or null. Missing values are not negative decisions.
A score strictly above the cutoff is positive, at/below is negative, overlapping
intervals abstain. These intervals bound incomplete evidence; they are not
calibrated confidence intervals. Unknown provenance abstains.

Independent: transfer, then raw q95, then RDSC, taking the first decisive answer.
Inherited: jointly available JAPD and squared-edge geometry, then RDSC. The dual
branch returns positive if either calibrated geometry component is positive,
but requires both components to be decisive. A decisive preferred answer is
preserved. The internal historical key `primary` means the No-Q ablation;
public `compose` and sequence field `basc_c` identify complete BASC-C.
