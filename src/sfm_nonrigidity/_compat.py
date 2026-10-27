"""Compatibility module aliases; canonical implementation lives in subpackages.

Aliases refer to the same module objects, never independent numerical copies.
The optional M3C2 dependency is loaded only on explicit request.
"""
from importlib import import_module
import sys

LEGACY_MODULES = {
    "alignment": "common.alignment",
    "analytic": "local.analytic",
    "classical": "comparators.classical",
    "comparator_plugin": "comparators.comparator_plugin",
    "conditioned": "local.conditioned",
    "controls": "methods.rdsc.controls",
    "edges": "local.edges",
    "epc_scalar": "methods.rdsc.epc_scalar",
    "geometry": "local.geometry",
    "global_model": "methods.global_modal.global_model",
    "global_projection": "methods.global_modal.global_projection",
    "identity": "local.identity",
    "intervals": "local.intervals",
    "linear": "common.linear",
    "linearization": "common.linearization",
    "m3c2": "comparators.m3c2",
    "models": "common.models",
    "precision": "common.precision",
    "qr_response": "common.qr_response",
    "rdsc": "methods.rdsc.rdsc",
    "routing": "methods.basc_c.routing",
    "sequence": "methods.basc_c.sequence",
    "transfer": "local.transfer",
    "transfer_views": "local.transfer_views"
}


def install(namespace):
    for old, canonical in LEGACY_MODULES.items():
        if old == "m3c2":
            continue
        module = import_module("." + canonical, "sfm_nonrigidity")
        sys.modules["sfm_nonrigidity." + old] = module
        namespace[old] = module
