"""Canonical and compatibility imports must share mutable numerical state."""
import importlib
import subprocess
import sys
import unittest
from unittest.mock import patch
from sfm_nonrigidity._compat import LEGACY_MODULES


class LayoutTests(unittest.TestCase):
    def test_legacy_imports_are_the_canonical_module_objects(self):
        for name, canonical in LEGACY_MODULES.items():
            if name == 'm3c2':
                continue  # optional native dependency, never loaded by core imports
            with self.subTest(module=name):
                self.assertIs(importlib.import_module('sfm_nonrigidity.' + name),
                              importlib.import_module('sfm_nonrigidity.' + canonical))

    def test_projection_callback_state_is_shared_across_import_paths(self):
        from sfm_nonrigidity import models, geometry
        from sfm_nonrigidity.common import models as canonical
        sentinel = object()
        with patch.object(canonical, 'project', sentinel):
            self.assertIs(models.project, sentinel)
            self.assertIs(geometry.L.P.G.project, sentinel)

    def test_core_import_does_not_load_optional_native_comparator(self):
        subprocess.run([sys.executable, '-c',
                        "import sys,sfm_nonrigidity; assert 'py4dgeo' not in sys.modules; "
                        "assert 'sfm_nonrigidity.comparators.m3c2' not in sys.modules"], check=True)


if __name__ == '__main__':
    unittest.main()
