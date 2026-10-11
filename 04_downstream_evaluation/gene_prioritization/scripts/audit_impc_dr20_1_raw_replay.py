from __future__ import annotations

import argparse
import json

from cellphenorank.impc_dr20 import audit_raw_replay_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit public raw-input coverage for the IMPC DR20.1 test units"
    )
    parser.add_argument("--eligible-cohort", required=True)
    parser.add_argument("--assertions", required=True)
    parser.add_argument("--hpoa", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = audit_raw_replay_inputs(
        eligible_cohort=args.eligible_cohort,
        assertions=args.assertions,
        hpoa=args.hpoa,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
