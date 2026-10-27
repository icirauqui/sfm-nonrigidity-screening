"""Verify maintained source identities; numerical reference checks are separate."""

from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "provenance/source_files.json").read_text())
    for name, expected in manifest.items():
        file = (root / name).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            parser.exit(1, f"Missing or invalid source path: {name}\n")
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            parser.exit(1, f"Source checksum mismatch: {name}\n")
    print(f"PASS: {len(manifest)} maintained source files")


if __name__ == "__main__":
    main()
