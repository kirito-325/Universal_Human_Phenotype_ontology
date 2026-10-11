from __future__ import annotations

import argparse
import json

from .baseline import reproduce_baselines
from .config import DEFAULT_ADAPTER, DEFAULT_REGISTRY, DEFAULT_RESOURCES, load_configs
from .resources import audit_resources, freeze_inputs


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="semantic_cpo")
    value.add_argument("--resources", default=str(DEFAULT_RESOURCES))
    value.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    value.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit-resources")
    commands.add_parser("reproduce-baselines")
    commands.add_parser("freeze-inputs")
    commands.add_parser("encode")
    stage = commands.add_parser("run-stage")
    stage.add_argument("--stage", required=True, choices=("nodes", "atoms", "profiles", "fusion"))
    commands.add_parser("run-controls")
    commands.add_parser("report")
    commands.add_parser("freeze-method")
    external = commands.add_parser("external-test")
    external.add_argument("--cohort")
    return value


def main() -> None:
    args = parser().parse_args()
    resources, registry, adapter = load_configs(args.resources, args.registry, args.adapter)
    if args.command == "audit-resources":
        result = audit_resources(resources, registry, adapter)
    elif args.command == "reproduce-baselines":
        result = reproduce_baselines(resources)
    elif args.command == "freeze-inputs":
        result = freeze_inputs(resources, registry, adapter)
    elif args.command in {"encode", "run-stage", "run-controls", "report", "freeze-method"}:
        from .stages import run_command

        result = run_command(args.command, resources, registry, adapter, getattr(args, "stage", None))
    elif args.command == "external-test":
        if not args.cohort:
            raise ValueError("external-test requires a separately supplied --cohort")
        from .stages import external_test

        result = external_test(resources, args.cohort)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

