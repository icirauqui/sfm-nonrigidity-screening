"""Global modal, RDSC and BASC-C research reference implementations."""

from .api import (
    global_scores,
    local_scores,
    rdsc_scores,
    epc_scores,
    compose,
    load_model,
    refine_points,
)

from ._compat import install as _install_legacy_aliases

_install_legacy_aliases(globals())
del _install_legacy_aliases

__version__ = "0.1.2"
__all__ = [
    "global_scores",
    "local_scores",
    "rdsc_scores",
    "epc_scores",
    "compose",
    "load_model",
    "refine_points",
]
