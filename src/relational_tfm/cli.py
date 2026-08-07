from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from relational_tfm.config import ConfigError, load_study_config
from relational_tfm.plan import atomic_write_json, build_plan, read_plan
from relational_tfm.preflight import run_preflight
from relational_tfm.provenance import dump_runtime_state
from relational_tfm.runner import run_shard, verify_shard
from relational_tfm.stage import StagingError, stage_manifest


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtfm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="expand and freeze a study plan")
    plan_parser.add_argument("--config", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify-plan", help="verify an immutable plan")
    verify_parser.add_argument("--plan", type=Path, required=True)
    verify_parser.add_argument("--expected-sha256")
    verify_parser.add_argument("--expected-cells", type=int)

    preflight_parser = subparsers.add_parser("preflight", help="validate a worker environment")
    preflight_parser.add_argument("--require-gpu", action="store_true")
    preflight_parser.add_argument("--require-github-push", action="store_true")

    runtime_parser = subparsers.add_parser("runtime-manifest", help="record runtime provenance")
    runtime_parser.add_argument("--output", type=Path, required=True)
    runtime_parser.add_argument("--plan", type=Path)

    stage_parser = subparsers.add_parser("stage-data", help="stage checksummed data assets")
    stage_parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/external_assets.yaml")
    )
    stage_parser.add_argument("--asset", action="append", dest="assets")
    stage_parser.add_argument("--force", action="store_true")

    shard_parser = subparsers.add_parser("run-shard", help="execute one deterministic plan shard")
    shard_parser.add_argument("--plan", type=Path, required=True)
    shard_parser.add_argument("--shard-index", type=int, required=True)
    shard_parser.add_argument("--cells-per-shard", type=int, required=True)
    shard_parser.add_argument("--output-dir", type=Path, required=True)

    verify_shard_parser = subparsers.add_parser(
        "verify-shard", help="verify a completed result shard"
    )
    verify_shard_parser.add_argument("--directory", type=Path, required=True)
    return parser


def command_plan(config_path: Path, output: Path) -> int:
    config = load_study_config(config_path)
    plan = build_plan(config)
    atomic_write_json(output, plan)
    print(f"study_id={plan['study_id']}")
    print(f"cells={len(plan['cells'])}")
    print(f"plan_sha256={plan['plan_sha256']}")
    print(f"output={output}")
    return 0


def command_verify_plan(
    path: Path, expected_sha256: str | None = None, expected_cells: int | None = None
) -> int:
    plan = read_plan(path)
    if expected_sha256 is not None and plan["plan_sha256"] != expected_sha256:
        raise ValueError(
            f"generated plan hash mismatch: expected {expected_sha256}, "
            f"computed {plan['plan_sha256']}"
        )
    if expected_cells is not None and len(plan["cells"]) != expected_cells:
        raise ValueError(
            f"generated plan cell count mismatch: expected {expected_cells}, "
            f"computed {len(plan['cells'])}"
        )
    print(f"PASS plan_sha256={plan['plan_sha256']} cells={len(plan['cells'])}")
    return 0


def command_preflight(require_gpu: bool, require_github_push: bool) -> int:
    checks = run_preflight(repository_root(), require_gpu, require_github_push)
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{status}\t{check.name}\t{check.detail}")
    return 0 if all(check.passed for check in checks) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            return command_plan(args.config, args.output)
        if args.command == "verify-plan":
            return command_verify_plan(args.plan, args.expected_sha256, args.expected_cells)
        if args.command == "preflight":
            return command_preflight(args.require_gpu, args.require_github_push)
        if args.command == "runtime-manifest":
            dump_runtime_state(repository_root(), args.output, args.plan)
            print(f"output={args.output}")
            return 0
        if args.command == "stage-data":
            results = stage_manifest(
                repository_root(), args.manifest, selected=set(args.assets or []), force=args.force
            )
            for result in results:
                print(f"{result['status']}\t{result['asset_id']}\t{result['target']}")
            return 0
        if args.command == "run-shard":
            manifest = run_shard(
                args.plan,
                args.shard_index,
                args.cells_per_shard,
                args.output_dir,
                repo_root=repository_root(),
            )
            print(
                f"PASS shard={manifest['shard_index']} cells={len(manifest['cell_ids'])} "
                f"manifest_sha256={manifest['manifest_sha256']}"
            )
            return 0
        if args.command == "verify-shard":
            manifest = verify_shard(args.directory)
            print(
                f"PASS shard={manifest['shard_index']} cells={len(manifest['cell_ids'])} "
                f"manifest_sha256={manifest['manifest_sha256']}"
            )
            return 0
    except (
        ConfigError,
        FileNotFoundError,
        ValueError,
        StagingError,
        json.JSONDecodeError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
