# Source entry points

`sfm_nonrigidity/api.py` is the public Python facade; `cli.py` and `__main__.py`
are the main executable entry points. Keep these thin and place implementations
in the method/component subpackages. The enclosing package under `src/` is the
installable Python namespace, so source checkout and wheel installs behave alike.

- `methods/global_modal/`: global-only modal projection and scoring.
- `methods/rdsc/`: original RDSC, coherent controls and exact EPC scalar path.
- `methods/basc_c/`: camera-provenance composition and temporal decisions.
- `local/`: local geometry, JAPD, edges and transfer components.
- `common/`: model IO, alignment and shared linear-response primitives.
- `comparators/`: elementary controls and optional published-method adapters.

See [architecture](../docs/architecture.md) for dependencies and compatibility.
