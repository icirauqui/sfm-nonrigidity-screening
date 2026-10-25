"""Explicit adapter for independently obtained optional comparator code.

Upstream LOG source has no verified redistribution license in the retained
snapshot and is not bundled. This loader neither downloads nor relicenses it.
"""

from pathlib import Path
import hashlib, importlib.util


def load_comparator(path, *, expected_sha256):
    path = Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("comparator_source_hash_mismatch")
    spec = importlib.util.spec_from_file_location("external_screening_comparator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "compare", None)):
        raise ValueError("comparator_requires_compare_function")
    return module.compare
