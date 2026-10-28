# Attribution and redistribution boundaries

The reference kernels were developed by Íñigo Cirauqui-Viloria in the Bundles /
MeshMeld research workspace. Camera-first similarity alignment and track pairing
are adapted from the author's GPL-3.0-or-later MeshMeld implementation. The
retained numerical definitions are preserved in this GPL-3.0-or-later package;
`provenance/kernel_equivalence.json` records numerical-function AST identities.
AI-assisted development and refactoring were reviewed through retained numerical
and source-equivalence checks; this is not independent external peer review.

NumPy, SciPy and optional Matplotlib, Pillow, PyMuPDF and py4dgeo are dependencies,
not copied upstream source trees. Their own licenses/attributions continue to
apply. The optional `m3c2` module is the author's adapter around native py4dgeo
M3C2 with independently qualified PCA normal selection and explicit sequence
support. It is not a full reproduction of physical covariance/significance.

The retained LOG upstream snapshot did not have a verified redistribution license.
Neither that source nor its source-derived lightweight graph implementation is
included here. `comparator_plugin.load_comparator` can load an independently
obtained, explicitly hash-checked module exposing `compare(x,y)`. The closed LOG
comparison results remain inspectable; this package does not claim a freely
redistributable full LOG implementation. Source permissions must be established
before redistributing such an implementation. This does not affect the author's
three proposed methods, elementary controls or native M3C2 adapter.

Display-only source geometry, states and RGB selections come from the Synthetic
Deformable Tubular Benchmark, DOI 10.57967/hf/10850, scientific version 1.0.0,
public naming edition v1.0.1. They retain their CC BY 4.0 attribution and are
kept in a separate input bundle, isolated from inference. Model display arrays
and closed numerical outputs are derived author research material. Software
licensing does not replace source-data licensing. The separate benchmark
reproduction resource DOI 10.57967/hf/10851 identifies a different resource.

No third-party raw clinical imagery, patient information, privileged truth input
for detector inference or reserved recovery-family payload is included.

The selected primary/artwork data deposit is separately licensed CC BY 4.0,
including its author-derived data/contract/result records; its explicit data
license does not extend to source software or optional dependencies. See the
fixed dataset identifiers and license link in `provenance/release_assets.json`.
