"""Command-line entry point for the dependency-free Phase 0 check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bootstrap import check_repository


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the OmniSearch Phase 0 scaffold."
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="Repository root to check."
    )
    args = parser.parse_args()

    report = check_repository(args.root)
    output = {
        "ok": report.ok,
        "config": {
            "name": report.config.name,
            "schema_version": report.config.schema_version,
            "seed": report.config.seed,
            "dataset_id": report.config.dataset_id,
            "split_group": report.config.split_group,
        },
        "missing_directories": list(report.missing_directories),
    }
    print(json.dumps(output, indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
