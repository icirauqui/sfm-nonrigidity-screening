# Repository contents and public release preparation

The public GitHub source repository is usable independently of the MeshMeld workspace.
It contains the core global/RDSC/BASC-C implementations, elementary controls,
point fitting, local components, calibrated composition and abstentions; public
API/CLI; installation pins; a manufactured six-command example; numerical tests;
compact hash-bound references; primary replay, result recount and artwork tools;
license/citation/third-party notices; source provenance; and CI/package recipes.
Use [usage.md](usage.md), [inputs.md](inputs.md), [reproduction.md](reproduction.md),
[numerics.md](numerics.md) and [development.md](development.md) as the reading route.

This is not the native MeshMeld application or an SfM frontend. It consumes
existing models/point exports. The methods retain the manuscript's bounded
scientific scope, failed utility gates and abstentions.

## What a Git checkout supplies

- Core inference without private workspace imports.
- Arithmetic examples that require no external inputs.
- Closed projected matrices/scores/predictions for compact replay and recount.
- Source for full primary model-to-score and display-only figure replay.
- A versioned dependency/environment record and explicit backend limitations.

The selected primary and artwork bundles are separate from source Git and public
at [the fixed input deposit](https://huggingface.co/datasets/icirauqui/sfm-nonrigidity-screening-reproduction/tree/0621f69360f78f5379afffb872532648d1e6e85a), version 0.1.0, revision `0621f69360f78f5379afffb872532648d1e6e85a`.
Names, sizes, hashes and fixed download URLs are in
[the asset index](../release_assets/README.md) and `provenance/release_assets.json`.
The deposit has CC BY 4.0 licensing, attribution and extraction/replay instructions.
DOI [10.57967/hf/11298](https://doi.org/10.57967/hf/11298) identifies this selected deposit. The benchmark dataset/support DOIs identify other resources.
Earlier private capsules, raw historical imagery and unlicensed LOG source are
not substitutes for this selected deposit.

## Release and validation boundaries

Source visibility, input publication, numerical replay and scientific validity
are distinct checks. Preserve exact source/data revisions and all numerical
refusals. Verify anonymous downloads, hashes/CRC, installation, compact checks,
full selected-input replay and artwork without replacing expected values or
loosening tolerances. See [numerics.md](numerics.md) for known cross-backend
strict-score differences despite decision agreement.

The input version supplies the stated primary model-to-score and display replay;
it does not supply every historical cohort, private upstream comparator,
original timing runtime or deleted RGB. Data version 0.1.0 is independent of
software version 0.1.2. The verified selected-input DOI is 10.57967/hf/11298; add an article identifier only when genuine. Source/data publication does not submit the article to a journal,
guarantee acceptance or establish clinical validity.
