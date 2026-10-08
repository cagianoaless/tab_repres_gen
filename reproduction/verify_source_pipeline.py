"""CPU-only source identity and representation checks; no model inference."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "pipeline"
RFMLINK = PIPELINE / "RFMLink"
HYBRID = RFMLINK / "side_experiments/g2t_q512_hybrid_solver_metric_free"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_file(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_global() -> ModuleType:
    # Original scripts import "protocol" by its CLI-local name. Isolate that
    # name so the verified implementation can also be used in the test suite.
    previous = sys.modules.get("protocol")
    sys.modules["protocol"] = load_file("_paper_global_protocol", HYBRID / "protocol.py")
    try:
        return load_file("_paper_global_features", HYBRID / "features.py")
    finally:
        if previous is None:
            sys.modules.pop("protocol", None)
        else:
            sys.modules["protocol"] = previous


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment: {path.name}/{name}")


def verify() -> dict:
    manifest = json.loads((HERE / "SOURCE_MANIFEST.json").read_text())
    seen = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        assert not relative.is_absolute() and ".." not in relative.parts, relative
        assert relative not in seen, f"duplicate source identity: {relative}"
        seen.add(relative)
        path = HERE / relative
        assert path.stat().st_size == item["bytes"], f"source size changed: {relative}"
        assert file_hash(path) == item["sha256"], f"source hash changed: {relative}"
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(relative))
    actual = {
        path.relative_to(HERE)
        for name in ("pipeline", "study", "reference")
        for path in (HERE / name).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert actual == seen, "source manifest does not cover the complete included closure"
    for relative, digest in {
        "rfmlink/vectorized_invariants.py":
            "d63ad3d0b8d3d1cb3d9cc1e160f66de9d276016e950f3dbf6b9b347d1e9dfb4d",
        "side_experiments/g2t_q512_hybrid_solver_metric_free/features.py":
            "d00aeb2e467d50649d326bc5dc384c7abaefc1aae5b7ab84abb2c0f89e9f0dda",
        "side_experiments/g2t_q512_eigengap_safe/features.py":
            "b21ff322409aefe256845272a1e5131254348c2253f99142c53da307c3e14194",
    }.items():
        assert file_hash(RFMLINK / relative) == digest

    local = load_file("_paper_local_features", RFMLINK / "rfmlink/vectorized_invariants.py")
    global13 = load_global()
    exporter = HERE / "study/scripts/export_rfmlink_source_panel_q64.py"
    b0 = literal_assignment(exporter, "B0_FEATURE_NAMES")
    b2 = literal_assignment(exporter, "B2V2_FEATURE_NAMES")
    assert tuple(local.FEATURE_NAMES) == b0[:18]
    assert b0[18:] == ("ultra_score",)
    assert tuple(global13.BASE_FEATURE_NAMES) == b0
    assert tuple(global13.B2_FEATURE_NAMES) == b2
    assert (len(local.FEATURE_NAMES), len(b0), len(b2), len(b0) + len(b2)) == (18, 19, 13, 32)
    assert global13.HEAT_SCALES == (0.1, 1.0, 10.0, 100.0)
    assert global13.LAPPE_DIMENSIONS == 8
    assert global13.NUM_FOLDS == 10
    assert (global13.DEGENERACY_RTOL, global13.DEGENERACY_ATOL) == (1e-5, 1e-7)
    assert global13.HYBRID_SOLVER_POLICY == {
        "dense_component_limit": 512,
        "large_component_solver": "scipy.sparse.linalg.eigsh",
        "which": "SM",
        "requested_eigenpairs": 10,
        "tolerance": 1e-10,
        "minimum_maxiter": 10000,
        "maxiter_per_node": 20,
        "residual_limit": 5e-5,
        "zero_eigenvalue_tolerance": 1e-8,
        "orthogonality_tolerance": 1e-8,
    }
    frozen = json.loads((HERE / "reference/frozen_numeric_protocol.json").read_text())
    implementation = global13.b2_implementation_sha256()
    assert implementation == frozen["b2_implementation_sha256"]
    assert implementation == "da165b3f24a8e8c9f9b5ed840400301bc30f30134274ab10c947ff32e76ed470"
    for name, digest in frozen["source_hashes"].items():
        assert file_hash(HYBRID / name) == digest
    reference = json.loads((HERE / "REFERENCE_EXECUTION.json").read_text())
    assert reference["ultra"]["checkpoint"] == "ultra_4g.pth"
    assert reference["ultra"]["commit"] == "427966ad8ed60420eef034063d44f3153addff90"
    assert reference["tabpfn"]["checkpoint"] == "tabpfn-v3-classifier-v3_default.ckpt"
    assert reference["tabpfn"]["version"] == "8.1.0"
    base = RFMLINK / "scripts/run_ultra4g_ood_reassessment.py"
    assert literal_assignment(base, "EXPECTED_ULTRA_SHA256") == reference["ultra"][
        "checkpoint_sha256"
    ]
    assert literal_assignment(base, "EXPECTED_TABPFN_SHA256") == reference["tabpfn"][
        "checkpoint_sha256"
    ]
    assert reference["broad_suite_allowed"] is True
    expected_plans = {
        "relation_context_cpu":
            "bd836b248fdb2812206087fae0c5ee88f5e9035cb9450c5d6d5da7893e9ccee2",
        "relation_context_tabpfn":
            "09461de58b61803baad894e8d3fb4e1da3980c13410fe36decac195c43553b7a",
        "gru":
            "bded7fc607c0fff7036f617599f721d3954d18f0d24df506dd94a2683bf686f5",
    }
    for key, digest in expected_plans.items():
        assert reference[key]["scientific_plan_sha256"] == digest
    for key, count in (
        ("relation_context_cpu", 2736), ("relation_context_tabpfn", 1368), ("gru", 15),
    ):
        assert reference[key]["cells"] == count
        assert len(reference[key]["scientific_plan_sha256"]) == 64
        assert len(reference[key]["serialized_file_sha256"]) == 64
    engine = ast.parse((RFMLINK / "scripts/run_ood_low_shot_confirmation.py").read_text())
    declaration = next(
        node for node in engine.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "FEATURE_NAMES" for t in node.targets)
    )
    assert ast.unparse(declaration.value) == "[*INVARIANT_NAMES, 'ultra_score']"
    return {"source_files": len(seen), "Local": 18, "score": 1, "Global": 13, "Full": 32}


def main() -> None:
    print("Source pipeline verification passed: " + json.dumps(verify(), sort_keys=True))


if __name__ == "__main__":
    main()
