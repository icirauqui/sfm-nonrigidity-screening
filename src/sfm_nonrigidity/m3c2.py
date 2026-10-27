"""Backward-compatible alias for the optional native M3C2 adapter."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module(".comparators.m3c2", "sfm_nonrigidity")
