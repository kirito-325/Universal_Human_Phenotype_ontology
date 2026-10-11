from __future__ import annotations

import argparse
import json

from cellphenorank.impc_dr20 import prepare_raw_replay_profiles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build middle-layer-independent DR20.1 HPO and mouse-model profiles"
    )
    parser.add_argument("--eligible-cohort", required=True)
    parser.add_argument("--candidate-universe", required=True)
    parser.add_argument("--assertions", required=True)
    parser.add_argument("--hpoa", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = prepare_raw_replay_profiles(
        eligible_cohort=args.eligible_cohort,
        candidate_universe=args.candidate_universe,
        assertions=args.assertions,
        hpoa=args.hpoa,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
