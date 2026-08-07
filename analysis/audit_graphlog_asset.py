from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

from relational_tfm.plan import atomic_write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = REPO_ROOT / "data/raw/graphlog/graphlog_v1.1.zip"
EXPECTED_SHA256 = "1788b5bbfaf9caf553db15437aa053c9a3810fbaefad9bd7b76a412b15122673"
CONTEXT_LENGTHS = (2, 3)
EVALUATION_LENGTHS = (4, 5)
CONTEXT_BUDGETS_PER_CLASS = (5, 10, 20, 40, 80)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(archive: zipfile.ZipFile, member: str) -> Iterable[dict[str, Any]]:
    with archive.open(member) as raw:
        for line_number, line in enumerate(raw, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{member}:{line_number} is invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{member}:{line_number} is not a JSON object")
            yield value


def _world_members(archive: zipfile.ZipFile) -> dict[str, dict[str, str]]:
    worlds: dict[str, dict[str, str]] = {}
    for member in archive.namelist():
        parts = Path(member).parts
        if len(parts) != 4 or parts[0] != "graphlog_v1.1":
            continue
        outer_split, world_name, filename = parts[1], parts[2], parts[3]
        if filename not in {"train.jsonl", "valid.jsonl", "test.jsonl"}:
            continue
        if not world_name.startswith("rule_"):
            raise ValueError(f"invalid GraphLog world directory: {world_name}")
        world = worlds.setdefault(
            world_name,
            {"outer_split": outer_split},
        )
        if world["outer_split"] != outer_split:
            raise ValueError(f"{world_name} appears in multiple outer splits")
        inner_split = filename.removesuffix(".jsonl")
        if inner_split in world:
            raise ValueError(f"{world_name} repeats inner split {inner_split}")
        world[inner_split] = member
    required = {"outer_split", "train", "valid", "test"}
    invalid = {
        world: sorted(required - set(members))
        for world, members in worlds.items()
        if set(members) != required
    }
    if invalid:
        raise ValueError(f"GraphLog worlds have incomplete split files: {invalid}")
    if len(worlds) != 57:
        raise ValueError(f"expected 57 GraphLog worlds, found {len(worlds)}")
    return worlds


def _descriptor_tokens(row: dict[str, Any]) -> tuple[str, ...]:
    descriptor = row.get("descriptor")
    if not isinstance(descriptor, str) or not descriptor:
        raise ValueError("GraphLog row has no descriptor")
    tokens = tuple(descriptor.split(","))
    if any(not token for token in tokens):
        raise ValueError("GraphLog descriptor contains an empty relation")
    return tokens


def _audit_row(row: dict[str, Any]) -> tuple[int, str, str, bool]:
    edges = row.get("edges")
    query = row.get("query")
    target = row.get("target")
    resolution_path = row.get("resolution_path")
    if (
        not isinstance(edges, list)
        or not isinstance(query, list)
        or len(query) != 3
        or not isinstance(target, str)
        or not isinstance(resolution_path, list)
    ):
        raise ValueError("GraphLog row omits its graph, query, target, or resolution path")
    tokens = _descriptor_tokens(row)
    if len(resolution_path) != len(tokens) + 1:
        raise ValueError("GraphLog descriptor and resolution path lengths disagree")
    if (query[0], query[1]) != (resolution_path[0], resolution_path[-1]):
        raise ValueError("GraphLog query endpoints disagree with its resolution path")
    if query[2] != target:
        raise ValueError("GraphLog query relation and target disagree")

    edge_relations: dict[tuple[Any, Any], set[str]] = {}
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 3:
            raise ValueError("GraphLog edge is not a typed triple")
        edge_relations.setdefault((edge[0], edge[1]), set()).add(str(edge[2]))
    descriptor_matches_graph_path = True
    for index, token in enumerate(tokens):
        pair = (resolution_path[index], resolution_path[index + 1])
        if token not in edge_relations.get(pair, set()):
            descriptor_matches_graph_path = False
    return len(tokens), target, ",".join(tokens), descriptor_matches_graph_path


def _audit_split(
    archive: zipfile.ZipFile,
    member: str,
) -> dict[str, Any]:
    lengths: Counter[int] = Counter()
    labels: Counter[str] = Counter()
    length_labels: dict[int, Counter[str]] = {}
    descriptors: Counter[str] = Counter()
    descriptor_graph_path_matches = 0
    rows = 0
    for line_number, row in enumerate(_jsonl(archive, member), start=1):
        try:
            length, label, descriptor, descriptor_matches = _audit_row(row)
        except ValueError as error:
            raise ValueError(f"{member}:{line_number}: {error}") from error
        rows += 1
        lengths[length] += 1
        labels[label] += 1
        length_labels.setdefault(length, Counter())[label] += 1
        descriptors[descriptor] += 1
        descriptor_graph_path_matches += int(descriptor_matches)
    if rows == 0:
        raise ValueError(f"GraphLog split is empty: {member}")
    return {
        "rows": rows,
        "length_counts": {str(key): value for key, value in sorted(lengths.items())},
        "label_counts": dict(sorted(labels.items())),
        "length_label_counts": {
            str(length): dict(sorted(counts.items()))
            for length, counts in sorted(length_labels.items())
        },
        "unique_descriptors": len(descriptors),
        "singleton_descriptor_rows": sum(
            count for count in descriptors.values() if count == 1
        ),
        "descriptor_graph_path_exact_match_rows": descriptor_graph_path_matches,
        "descriptor_graph_path_exact_match_fraction": (
            descriptor_graph_path_matches / rows
        ),
    }


def build_audit(asset: Path) -> dict[str, Any]:
    actual_sha256 = file_sha256(asset)
    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"GraphLog asset checksum mismatch: expected {EXPECTED_SHA256}, "
            f"got {actual_sha256}"
        )
    with zipfile.ZipFile(asset) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"GraphLog ZIP integrity failed at {bad_member}")
        worlds = _world_members(archive)
        world_audits = []
        for world_name in sorted(worlds, key=lambda value: int(value.split("_")[1])):
            members = worlds[world_name]
            split_audits = {
                split: _audit_split(archive, members[split])
                for split in ("train", "valid", "test")
            }
            context_counts: Counter[str] = Counter()
            for length in CONTEXT_LENGTHS:
                context_counts.update(
                    split_audits["train"]["length_label_counts"].get(str(length), {})
                )
            evaluation_counts: Counter[str] = Counter()
            for length in EVALUATION_LENGTHS:
                evaluation_counts.update(
                    split_audits["test"]["length_label_counts"].get(str(length), {})
                )
            evaluation_labels = sorted(
                label for label, count in evaluation_counts.items() if count > 0
            )
            minimum_context = min(
                (context_counts[label] for label in evaluation_labels),
                default=0,
            )
            minimum_evaluation = min(
                (evaluation_counts[label] for label in evaluation_labels),
                default=0,
            )
            world_audits.append(
                {
                    "world": world_name,
                    "outer_split": members["outer_split"],
                    "splits": split_audits,
                    "short_context_labels": len(context_counts),
                    "long_evaluation_labels": len(evaluation_labels),
                    "minimum_short_context_rows_per_evaluation_label": minimum_context,
                    "minimum_long_evaluation_rows_per_label": minimum_evaluation,
                    "supported_context_budgets_per_class": [
                        budget
                        for budget in CONTEXT_BUDGETS_PER_CLASS
                        if minimum_context >= budget
                    ],
                    "all_long_evaluation_labels_seen_in_short_context": all(
                        context_counts[label] > 0 for label in evaluation_labels
                    ),
                }
            )

    outer_split_counts = Counter(audit["outer_split"] for audit in world_audits)
    budget_world_counts = {
        str(budget): sum(
            budget in audit["supported_context_budgets_per_class"]
            for audit in world_audits
        )
        for budget in CONTEXT_BUDGETS_PER_CLASS
    }
    return {
        "schema_version": 1,
        "status": "PASS",
        "asset": {
            "path": str(asset.relative_to(REPO_ROOT)),
            "sha256": actual_sha256,
            "official_repository_commit": "03bf50bd54664d48db2c0e272354f636d6c2198f",
            "official_repository_url": "https://github.com/facebookresearch/GraphLog",
            "author_mirror_issue": "https://github.com/facebookresearch/GraphLog/issues/34",
            "author_mirror_file_id": "1s6oG_5Ul199puKAu67z3QrOAm943hGV2",
            "archive_root": "graphlog_v1.1",
        },
        "design_probe": {
            "context_inner_split": "train",
            "evaluation_inner_split": "test",
            "context_lengths": list(CONTEXT_LENGTHS),
            "evaluation_lengths": list(EVALUATION_LENGTHS),
            "candidate_context_budgets_per_class": list(
                CONTEXT_BUDGETS_PER_CLASS
            ),
            "worlds_supporting_each_budget": budget_world_counts,
        },
        "world_count": len(world_audits),
        "outer_split_counts": dict(sorted(outer_split_counts.items())),
        "worlds": world_audits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = build_audit(args.asset.resolve())
    if args.output is not None:
        atomic_write_json(args.output, audit)
    compact = {
        "status": audit["status"],
        "asset_sha256": audit["asset"]["sha256"],
        "world_count": audit["world_count"],
        "outer_split_counts": audit["outer_split_counts"],
        "worlds_supporting_each_budget": audit["design_probe"][
            "worlds_supporting_each_budget"
        ],
    }
    print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
