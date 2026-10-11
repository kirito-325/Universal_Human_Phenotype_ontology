from __future__ import annotations

import argparse
import json

from cellphenorank.impc_dr20 import prepare_impc_dr20_reproduction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the official IMPC DR20.1/2024 benchmark"
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = prepare_impc_dr20_reproduction(args.source_dir, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

