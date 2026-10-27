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

The selected primary and artwork bundles are intentionally separate from Git.
Their names, sizes and hashes are in [the asset index](../release_assets/README.md)
and `provenance/release_assets.json`. They are prepared and verified locally but
are not yet publicly downloadable. Until they are deposited, a public reader
cannot perform the full model/artwork replay from a Git clone alone. The compact
check remains available. Earlier private capsules, raw historical imagery and
unlicensed LOG source are not substitutes for these selected bundles.

## Author-controlled public release steps

1. Source visibility is public (verified 27 October 2026). Preserve the
   exact source commit and identify the intended software version. Do not overwrite historical numerical reference records.
2. Deposit both selected input bundles in a suitable versioned resource, keeping
   model inference and display-only inputs separate. Include checksums, original
   source-data licensing/attribution and extraction instructions.
3. Replace the null public asset URL/DOI fields in
   `provenance/release_assets.json` with verified identifiers. Document the exact
   version/URLs and update reproduction instructions. Do not reuse the benchmark
   dataset/support DOIs as identifiers for this diagnostic implementation.
4. Verify anonymous access, download hashes/CRC, installation, compact checks,
   full primary replay and artwork from the deposited bytes in a fresh checkout.
   Keep every numerical discrepancy and distinguish source/decision equivalence
   from strict historic score parity. Qualify any changed runtime separately.
5. Update the article's code/data statement and software citation to identify
   the actual deposit/version. Add the screening article identifier only when
   genuine; the companion benchmark preprint is a separate citation.
6. Update the workspace gitlink and dependency lock after any source commit,
   then verify its import and fresh-submodule initialization. Preserve ordinary
   commit history; no force push or replacement of published artifacts is needed.

Public source access and submodule registration do not complete the remaining
input-deposition and external-reproduction steps. Publication visibility and a DOI are not necessary
to run the core implementation, but public input access is needed for complete
external reproduction of the paper's primary figures and scores.

## Explicit remaining boundaries

Historic E16/J08 raw imagery is unavailable. Some cross-backend JAPD/edge scores
exceed the retained strict tolerance, even when decisions match. The LOG adapter
requires independently obtained, explicitly hash-qualified external code;
upstream source redistribution permission is unresolved. M3C2 needs its optional
native dependency/runtime. These constraints remain documented in the methods,
reproduction guide and third-party notice; none prevents using the three core
diagnostics within their declared input contracts.
