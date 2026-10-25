# Methods and evidence boundaries

`global_model` builds the fixed 24 affine-temporal and 36 quadratic-temporal
columns and signed-pixel residual blocks. `global_projection` eliminates point
and camera nuisance responses and measures the quadratic gain after affine
removal. `mode_gain` is unnormalized squared-pixel energy; the optional
`raw_pixel_F_stream_r0` studentizes the same gain. Primary energy and F decisions
tie. Observation support, resolution, residual scale and track multiplicity can
affect scores; the study does not establish support-size invariance or nominal
statistical significance. Rank thresholds, zero-energy refusals and df accounting
are inherited exactly from the evaluated implementation.

`rdsc` applies robust affine detrending and neighbor averaging. Let a be q95
coherent amplitude and b the median shuffled coherent amplitude. RDSC is
`a² / max(b, tiny)`. Elementary controls include a, a/b, a−b and **a²−b²**;
the last may be negative and is not (a−b)². Coherent amplitude matches RDSC
in the primary comparison. Failed complexity-specific utility gates are retained.
Affine deformation and poorly represented spatial modes can be removed or missed.

`geometry`, `analytic`, `conditioned`, `edges`, `linear` and `qr_response` supply
point-only fitting, shared pixel responses, JAPD and squared-edge contrasts.
JAPD is the median square-root regularized Mahalanobis norm, without division
by three. Pixel variance averages squared 2D vector residual norms over shared
observations, while its df denominator uses full unique local track length.
This is a conditional empirical proxy, not a proven physical noise distribution.

`routing` and `sequence` implement BASC-C's provenance branches and selective
calibration. Its RDSC fallback has bounded coverage value, but neither a universal
composition advantage nor a global-inclusive cascade is established.

Global and local geometry are estimated references. Their disagreements can
reflect estimator bias, correspondence, support, calibration and gauge effects.
The software performs screening, not physical deformation measurement or recovery.
The thesis-derived Bundles workflow predates this implementation. Projection and
nuisance removal use established numerical ideas; no first-of-kind general
change-detection theory is claimed. The associated article distinguishes these
origins and the contribution of the present comparative study.
