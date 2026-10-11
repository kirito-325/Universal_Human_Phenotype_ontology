from __future__ import annotations

import argparse
import json

from cellphenorank.config import load_config
from cellphenorank.impc_dr20_embedding import run_impc_dr20_embedding_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test HPO-MP embeddings against official DR20.1 gene-level outcomes"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_impc_dr20_embedding_test(
        load_config(args.config), args.output_dir
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

