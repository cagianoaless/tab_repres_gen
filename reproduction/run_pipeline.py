"""Portable paths around unchanged, checksummed paper producer entry points."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GLOBAL_SCRIPTS = (
    "spectral_audit",
    "cutoff_audit",
    "development_large_stability",
    "materialize",
    "finalize_numeric_protocol",
)
STUDY_SCRIPTS = (
    "export_rfmlink_source_panel_q64",
    "build_rfmlink_relation_context_asset",
    "build_rfmlink_relation_context_parallel",
    "generate_rfmlink_relation_context_registration",
)


def initialize(work: Path, ultra: Path, checkpoint: Path) -> None:
    from reproduction.verify_source_pipeline import file_hash, verify

    verify()
    ultra, checkpoint = ultra.resolve(), checkpoint.resolve()
    reference = json.loads((HERE / "REFERENCE_EXECUTION.json").read_text())
    head = subprocess.check_output(
        ["git", "-C", str(ultra), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != reference["ultra"]["commit"]:
        raise ValueError("ULTRA checkout differs from the pinned producer dependency")
    if subprocess.check_output(["git", "-C", str(ultra), "diff", "HEAD", "--"]):
        raise ValueError("ULTRA tracked source has local modifications")
    for path, expected in (
        (ultra / "ckpts/ultra_4g.pth", reference["ultra"]["checkpoint_sha256"]),
        (checkpoint, reference["tabpfn"]["checkpoint_sha256"]),
    ):
        if file_hash(path) != expected:
            raise ValueError(f"checkpoint identity mismatch: {path.name}")
    manifest = json.loads((HERE / "SOURCE_MANIFEST.json").read_text())
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.parts[0] != "pipeline":
            continue
        target = work / Path(*relative.parts[1:])
        if target.exists() and file_hash(target) != item["sha256"]:
            raise ValueError(f"refusing to overwrite modified workspace source: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(HERE / relative, target)
    external = work / "RFMLink/external/ULTRA"
    external.parent.mkdir(parents=True, exist_ok=True)
    if external.is_symlink() or external.exists():
        if external.resolve() != ultra:
            raise ValueError("workspace ULTRA path already refers to another checkout")
    else:
        external.symlink_to(ultra, target_is_directory=True)
    # This is a clearly labelled reference gate, not a fabricated new audit.
    gate = work / "reference/paper_execution_gate.json"
    gate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "REFERENCE_EXECUTION.json", gate)
    # The original CLI requires these fixed-policy reference numerical gates.
    # Preserve their bytes and identity, rather than presenting them as reruns.
    stability = (
        work / "results/analysis/g2t_q512_metric_free_large_eigsh_stability_v1/shards"
    )
    stability.mkdir(parents=True, exist_ok=True)
    for source in (HERE / "reference/large_component_stability").glob("shard*of2.json"):
        target = stability / source.name
        if target.exists() and file_hash(target) != file_hash(source):
            raise ValueError(f"workspace numerical reference differs: {target}")
        if not target.exists():
            shutil.copyfile(source, target)
    settings = work / "reproduction_paths.json"
    payload = {"ultra_root": str(ultra), "tabpfn_checkpoint": str(checkpoint)}
    if settings.exists() and json.loads(settings.read_text()) != payload:
        raise ValueError("workspace is already initialized with different dependencies")
    settings.write_text(json.dumps(payload, indent=2) + "\n")


def configure_producer(work: Path, stage: str):
    settings = json.loads((work / "reproduction_paths.json").read_text())
    checkpoint = Path(settings["tabpfn_checkpoint"])
    sys.path[:0] = [str(work / "RFMLink/scripts"), str(work / "RFMLink"), str(work)]
    paths = importlib.import_module("rfmlink.paths")
    paths.MODELS_CKPT = checkpoint.parent
    paths.TABPFN3_CLASSIFIER_CKPT = checkpoint
    paths.TFMS_PYTHON = Path(sys.executable)
    suite = importlib.import_module("run_ultra57_rfmlink")
    suite.base.TABPFN_CHECKPOINT = checkpoint
    suite.MASKING_AUDIT_PATH = work / "reference/paper_execution_gate.json"
    if stage == "q512":
        suite = importlib.import_module("run_ultra57_rfmlink_q512").suite
    return suite


def execute_script(script: Path, arguments: list[str]) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(HERE / "study"), str(REPO), env.get("PYTHONPATH", "")]
    )
    subprocess.run([sys.executable, str(script), *arguments], env=env, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    stages = parser.add_subparsers(dest="stage", required=True)
    init = stages.add_parser("init", help="verify dependencies and stage unchanged sources")
    init.add_argument("--ultra-root", type=Path, required=True)
    init.add_argument("--tabpfn-checkpoint", type=Path, required=True)
    for name in ("batch1", "q512"):
        stages.add_parser(name).add_argument("arguments", nargs=argparse.REMAINDER)
    glob = stages.add_parser("global", help="run the original numerical validation/materializer")
    glob.add_argument("--script", choices=GLOBAL_SCRIPTS, required=True)
    glob.add_argument("arguments", nargs=argparse.REMAINDER)
    study = stages.add_parser("study", help="run original export/context/registration CLI")
    study.add_argument("--script", choices=STUDY_SCRIPTS, required=True)
    study.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    work = args.work_root.resolve()
    if work == REPO or work.is_relative_to(HERE):
        raise ValueError("choose a generated workspace, not the repository/source directory")
    if args.stage == "init":
        initialize(work, args.ultra_root, args.tabpfn_checkpoint)
        return
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if args.stage in {"batch1", "q512"}:
        suite = configure_producer(work, args.stage)
        sys.argv = [f"{args.stage} producer", *arguments]
        suite.main()
    elif args.stage == "global":
        script = (
            work / "RFMLink/side_experiments/g2t_q512_hybrid_solver_metric_free"
            / f"{args.script}.py"
        )
        execute_script(script, arguments)
    else:
        execute_script(HERE / "study/scripts" / f"{args.script}.py", arguments)


if __name__ == "__main__":
    main()
