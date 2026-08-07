from analysis.summarize_clutrr_shared_tabpfn import (
    CONTEXTS,
    ENCODERS,
    OOD_LENGTHS,
    SEEDS,
    summarize,
)


def test_shared_tabpfn_summary_preserves_pairing() -> None:
    values = {
        "relation_transition_counts": 0.40,
        "within_length_transition_joint_permutation": 0.10,
        "ordered_plus_multiset": 0.35,
        "within_length_ordered_plus_multiset_joint_permutation": 0.15,
    }
    payloads = []
    for encoder in ENCODERS:
        for context in CONTEXTS:
            for seed in SEEDS:
                payloads.append(
                    {
                        "cell": {
                            "encoder": encoder,
                            "context_per_relation": context,
                            "seed": seed,
                        },
                        "metrics_by_chain_length": {
                            str(length): {"accuracy": values[encoder]}
                            for length in OOD_LENGTHS
                        },
                    }
                )

    tables = summarize(payloads)

    assert len(tables["cells"]) == 60
    assert len(tables["aggregate"]) == 20
    assert len(tables["contrasts"]) == 30
    largest = [
        row for row in tables["contrasts"] if row["context_per_relation"] == 80
    ]
    assert len(largest) == 6
    assert {
        (row["contrast"], round(row["difference"], 2)) for row in largest
    } == {
        ("transition_aligned_minus_null", 0.30),
        ("ordered_plus_multiset_aligned_minus_null", 0.20),
    }
