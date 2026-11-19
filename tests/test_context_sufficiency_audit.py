"""Synthetic invariants for the prospective context-only witness diagnostic."""

from types import SimpleNamespace

import pandas as pd

from analysis.audit_context_sufficiency import (
    aggregate,
    infer_productions,
    possible_roots,
    summarize_task,
    write_outputs,
)


def test_recursive_binary_derivation():
    rules, conflicts = infer_productions([(("a", "b"), "c"), (("c", "d"), "e")], 2)
    assert not conflicts
    assert possible_roots(("a", "b", "d"), rules, 2) == {"e"}
    assert possible_roots(("a", "b", "x"), rules, 2) == set()


def test_conflicting_short_descriptors_are_not_rules():
    rules, conflicts = infer_productions([(("a", "b"), "x"), (("a", "b"), "y")], 2)
    assert ("a", "b") not in rules
    assert len(conflicts[("a", "b")]) == 2


def test_ternary_rules_are_a_separate_sensitivity():
    pairs = [(("a", "b", "c"), "d"), (("d", "e"), "z")]
    binary, _ = infer_productions(pairs, 2)
    ternary, _ = infer_productions(pairs, 3)
    assert not possible_roots(("a", "b", "c", "e"), binary, 2)
    assert possible_roots(("a", "b", "c", "e"), ternary, 3) == {"z"}


def test_parse_reports_multiple_possible_roots():
    pairs = [(("a", "b"), "c"), (("c", "d"), "e"),
             (("b", "d"), "x"), (("a", "x"), "q")]
    rules, _ = infer_productions(pairs, 2)
    assert possible_roots(("a", "b", "d"), rules, 2) == {"e", "q"}


def test_ground_truth_does_not_change_generated_witnesses():
    context = pd.DataFrame([
        {"relation_tokens": ("a", "b"), "target_relation": "c"},
        {"relation_tokens": ("c", "d"), "target_relation": "e"},
    ])
    template = {"row_id": "query1", "relation_tokens": ("a", "b", "d"),
                "descriptor_graph_path_exact_match": True}
    first = SimpleNamespace(context=context, evaluation=pd.DataFrame([
        {**template, "target_relation": "e"}
    ]))
    second = SimpleNamespace(context=context, evaluation=pd.DataFrame([
        {**template, "target_relation": "q"}
    ]))
    a, aq = summarize_task(first, "graphlog", "rule_0", 101, 40)
    b, bq = summarize_task(second, "graphlog", "rule_0", 101, 40)
    assert aq[0]["arity2_roots"] == bq[0]["arity2_roots"] == "e"
    assert a["by_length"]["3"]["arity2_unique_correct"] == 1
    assert b["by_length"]["3"]["arity2_unique_wrong"] == 1
    assert a["by_length"]["3"]["descriptor_matches_graph_path"] == 1


def test_micro_and_equal_world_rates_separate():
    a = {"dataset": "graphlog", "world": "rule_0", "by_length": {
        "4": {"rows": 2, "arity2_unique_correct": 2}}}
    b = {"dataset": "graphlog", "world": "rule_1", "by_length": {
        "4": {"rows": 8, "arity2_unique_correct": 0}}}
    result = aggregate([a, b])["by_length"][0]
    assert result["micro_rates"]["arity2_unique_correct"] == 0.2
    assert result["equal_world_seed_rates"]["arity2_unique_correct"] == 0.5


def test_outputs_generated_not_misrepresented_as_verified(tmp_path):
    a = {"dataset": "clutrr", "world": "single", "seed": 101,
         "context_per_relation": 80, "context_rows": 2,
         "by_length": {"4": {"rows": 1, "arity2_unique_correct": 1}}}
    report = write_outputs(tmp_path, [{"plan_sha256": "demo"}], [a], [], emit_queries=True)
    assert report["status"] == "COMPLETED_ON_CHECKSUM_VERIFIED_DATA"
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "cell_summary.csv").is_file()
    assert (tmp_path / "query_evidence.csv").is_file()
