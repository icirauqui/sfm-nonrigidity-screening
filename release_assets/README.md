# Prepared input assets

These ZIPs are built and checked locally; they are excluded from Git. Their
sizes/hashes are in ../provenance/release_assets.json and SHA256SUMS. Public
asset URLs and a deposit DOI remain unissued. The source repository must not
claim they are already downloadable.

- screening-primary-inputs-v0.1.0.zip: explicit model/point inputs and frozen
  calibration/window requests, no RGB/evaluator truth.
- screening-artwork-inputs-v0.1.0.zip: selected display-only benchmark geometry,
  RGB, model-display arrays and closed results, isolated from inference.

Extract each into a new directory; do not merge the two input roles. Source
benchmark attribution/license and reproduction commands are in ../THIRD_PARTY.md
and ../docs/reproduction.md. The author's private historical review capsule is
not one of these public-preparation assets and must not be uploaded in its place.
