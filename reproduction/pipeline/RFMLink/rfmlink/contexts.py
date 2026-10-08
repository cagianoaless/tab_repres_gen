"""Module 1 — Relational context module.

Samples labeled context triples (positives from train, negatives by head/tail
corruption) for the in-context decoder. Negative filtering is train-only by
default; the optional all-true filter is explicitly recorded in the config.

Output layout (spec-mandated):
    contexts/<dataset>/<seed>/<context_size>/context_triples.npy   int64 [n, 3]
    contexts/<dataset>/<seed>/<context_size>/context_labels.npy    int64 [n]
    contexts/<dataset>/<seed>/<context_size>/context_config.json

<context_size> is "<P>pos_<N>neg" or "full".
"""

import json

import numpy as np

from .paths import CONTEXTS_DIR

CONTEXT_SIZES = (50, 100, 200, 400)


def _corrupt_negatives(positives, num_entities, count, forbidden, rng):
    """Head-or-tail corruption of random positive triples, rejecting any
    candidate in `forbidden` (and duplicates)."""
    rows = []
    used = set()
    attempts = 0
    max_attempts = count * 200
    while len(rows) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"negative sampling failed: {len(rows)}/{count} after {attempts} attempts"
            )
        h, r, t = positives[rng.integers(0, len(positives))]
        if rng.random() < 0.5:
            h = int(rng.integers(0, num_entities))
        else:
            t = int(rng.integers(0, num_entities))
        candidate = (int(h), int(r), int(t))
        if candidate in forbidden or candidate in used:
            continue
        used.add(candidate)
        rows.append(candidate)
    return np.asarray(rows, dtype=np.int64)


def build_context(
    dataset,
    train_triples,
    num_entities,
    seed,
    num_positive,
    num_negative,
    all_true_triples=None,
    negative_filtering_mode="train_only",
):
    """Sample a labeled context. Returns (triples [n,3], labels [n], config)."""
    train_triples = np.asarray(train_triples, dtype=np.int64)
    train_set = {tuple(t) for t in train_triples.tolist()}
    rng = np.random.default_rng(seed)

    if negative_filtering_mode == "train_only":
        forbidden = train_set
        all_true_filter_used = False
    elif negative_filtering_mode == "all_true":
        if all_true_triples is None:
            raise ValueError("all_true filtering requested without all_true_triples")
        forbidden = set(all_true_triples)
        all_true_filter_used = True
    else:
        raise ValueError(f"unknown negative_filtering_mode {negative_filtering_mode!r}")

    if num_positive == "full":
        positive = train_triples
        num_positive_actual = len(train_triples)
    else:
        if num_positive > len(train_triples):
            raise ValueError(
                f"requested {num_positive} positives but train has {len(train_triples)}"
            )
        indices = rng.choice(len(train_triples), size=num_positive, replace=False)
        positive = train_triples[indices]
        num_positive_actual = num_positive

    if num_negative == "full":
        num_negative_actual = num_positive_actual
    else:
        num_negative_actual = num_negative
    negative = _corrupt_negatives(
        train_triples, num_entities, num_negative_actual, forbidden, rng
    )

    triples = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(positive), dtype=np.int64),
            np.zeros(len(negative), dtype=np.int64),
        ]
    )
    order = rng.permutation(len(triples))
    triples, labels = triples[order], labels[order]

    config = {
        "dataset": dataset,
        "seed": int(seed),
        "num_positive": int(num_positive_actual),
        "num_negative": int(num_negative_actual),
        "negative_filtering_mode": negative_filtering_mode,
        "all_true_filter_used": all_true_filter_used,
        "source_split": "train",
        "negative_sampling": "uniform head-or-tail corruption of random train triples, duplicate-rejected",
        "leakage_notes": (
            "positives sampled uniformly from train only; negatives rejected against "
            + ("train true triples only (valid/test positives may appear as negatives, standard KGC practice)"
               if not all_true_filter_used
               else "ALL true triples (train+valid+test) — uses valid/test information, explicitly flagged")
        ),
    }
    return triples, labels, config


def context_dir(dataset, seed, size_label):
    return CONTEXTS_DIR / dataset / str(seed) / size_label


def save_context(dataset, seed, size_label, triples, labels, config):
    out = context_dir(dataset, seed, size_label)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "context_triples.npy", triples)
    np.save(out / "context_labels.npy", labels)
    with (out / "context_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return out


def load_context(dataset, seed, size_label):
    out = context_dir(dataset, seed, size_label)
    triples = np.load(out / "context_triples.npy")
    labels = np.load(out / "context_labels.npy")
    with (out / "context_config.json").open(encoding="utf-8") as f:
        config = json.load(f)
    return triples, labels, config


def build_all_contexts(dataset, train_triples, num_entities, seeds, include_full=True):
    """Materialize the spec's context grid for one dataset."""
    written = []
    for seed in seeds:
        for size in CONTEXT_SIZES:
            triples, labels, config = build_context(
                dataset, train_triples, num_entities, seed, size, size
            )
            written.append(
                str(save_context(dataset, seed, f"{size}pos_{size}neg", triples, labels, config))
            )
        if include_full:
            triples, labels, config = build_context(
                dataset, train_triples, num_entities, seed, "full", "full"
            )
            written.append(
                str(save_context(dataset, seed, "full", triples, labels, config))
            )
    return written
