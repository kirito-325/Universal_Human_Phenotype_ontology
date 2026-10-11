from __future__ import annotations

import argparse
import json

from cpo_pipeline.config import load_config, validate_config
from uhpo_common.paths import PROFILE_SIMILARITY

from .disease_model import run_disease_model
from .evaluate import run_evaluate
from .gene_rank import run_gene_rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Downstream prediction using prepared CPO matrices")
    parser.add_argument("stage", choices=["disease-model", "gene-rank", "evaluate", "report"])
    parser.add_argument("--config", default=PROFILE_SIMILARITY / "config/cpo_v2.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    if args.stage == "disease-model":
        result = run_disease_model(config, resume=args.resume)
    elif args.stage == "gene-rank":
        result = run_gene_rank(config)
    elif args.stage == "evaluate":
        result = run_evaluate(config)
    else:
        from .report import run_report
        result = run_report(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
