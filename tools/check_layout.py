"""Check reorganized numerical bodies against the retained migration AST hashes."""
from pathlib import Path
import ast
import hashlib
import json
import sys


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use Python 3.12 to compare the retained Python 3.12 AST hashes")
    root = Path(__file__).resolve().parents[1]
    layout = json.loads((root / 'provenance/package_layout.json').read_text())['legacy_modules']
    reference = json.loads((root / 'provenance/kernel_equivalence.json').read_text())
    definitions = 0
    for record in reference['kernels']:
        path = root / 'src/sfm_nonrigidity' / (layout[record['module']].replace('.', '/') + '.py')
        actual = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                  for node in ast.parse(path.read_text()).body
                  if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        expected = {row['name']: row['extracted_ast_sha256'] for row in record['functions']}
        if actual != expected:
            raise SystemExit('Numerical definition mismatch: ' + str(path))
        definitions += len(actual)
    if definitions != reference['definitions']:
        raise SystemExit('Unexpected numerical definition count')
    print(f"PASS: {definitions} unchanged numerical definitions in {len(reference['kernels'])} relocated modules")


if __name__ == '__main__':
    main()
