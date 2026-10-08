"""Frozen registry for the 57 link-prediction datasets in ULTRA's README."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Ultra57Dataset:
    slug: str
    class_name: str
    version: str | None
    family: str
    target_validation: bool
    official_evaluation: str
    ultra4g_training_member: bool = False
    completed_in_ultra4g_reassessment: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _transductive(
    slug: str,
    class_name: str,
    *,
    tail_only: bool = False,
    training_member: bool = False,
    completed: bool = False,
) -> Ultra57Dataset:
    return Ultra57Dataset(
        slug=slug,
        class_name=class_name,
        version=None,
        family="transductive",
        target_validation=True,
        official_evaluation="tail_only" if tail_only else "head_and_tail",
        ultra4g_training_member=training_member,
        completed_in_ultra4g_reassessment=completed,
    )


def _inductive(
    slug: str,
    class_name: str,
    version: str,
    family: str,
    *,
    target_validation: bool,
) -> Ultra57Dataset:
    return Ultra57Dataset(
        slug=slug,
        class_name=class_name,
        version=version,
        family=family,
        target_validation=target_validation,
        official_evaluation="head_and_tail",
    )


TRANSDUCTIVE = (
    _transductive("fb15k237", "FB15k237", training_member=True),
    _transductive("wn18rr", "WN18RR", training_member=True),
    _transductive("nell995", "NELL995", training_member=True),
    _transductive("yago310", "YAGO310", completed=True),
    _transductive("codex_small", "CoDExSmall"),
    _transductive("codex_medium", "CoDExMedium", training_member=True),
    _transductive("codex_large", "CoDExLarge"),
    _transductive("hetionet", "Hetionet", completed=True),
    _transductive("conceptnet100k", "ConceptNet100k"),
    _transductive("dbpedia100k", "DBpedia100k"),
    _transductive("aristov4", "AristoV4", completed=True),
    _transductive("wdsinger", "WDsinger", tail_only=True),
    _transductive("nell23k", "NELL23k", tail_only=True),
    _transductive("fb15k237_10", "FB15k237_10", tail_only=True),
    _transductive("fb15k237_20", "FB15k237_20", tail_only=True),
    _transductive("fb15k237_50", "FB15k237_50", tail_only=True),
)


GRAIL = tuple(
    _inductive(
        f"{prefix}_{version}",
        class_name,
        version,
        "inductive_entity_grail",
        target_validation=True,
    )
    for prefix, class_name in (
        ("fb15k237_ind", "FB15k237Inductive"),
        ("wn18rr_ind", "WN18RRInductive"),
        ("nell_ind", "NELLInductive"),
    )
    for version in ("v1", "v2", "v3", "v4")
)


ILPC = tuple(
    _inductive(
        f"ilpc2022_{version}",
        "ILPC2022",
        version,
        "inductive_entity_ilpc",
        target_validation=True,
    )
    for version in ("small", "large")
)


HM = tuple(
    _inductive(
        f"hm_{version}",
        "HM",
        version,
        "inductive_entity_hm",
        target_validation=False,
    )
    for version in ("1k", "3k", "5k", "indigo")
)


INGRAM = tuple(
    _inductive(
        f"{prefix}_{version}",
        class_name,
        version,
        "inductive_entity_relation_ingram",
        target_validation=True,
    )
    for prefix, class_name, versions in (
        ("fb_ingram", "FBIngram", ("25", "50", "75", "100")),
        ("wk_ingram", "WKIngram", ("25", "50", "75", "100")),
        ("nl_ingram", "NLIngram", ("0", "25", "50", "75", "100")),
    )
    for version in versions
)


MTDEA = (
    *(
        _inductive(
            f"{prefix}_{version}",
            class_name,
            version,
            "inductive_entity_relation_mtdea",
            target_validation=False,
        )
        for prefix, class_name, versions in (
            ("wiki_mt1", "WikiTopicsMT1", ("tax", "health")),
            ("wiki_mt2", "WikiTopicsMT2", ("org", "sci")),
            ("wiki_mt3", "WikiTopicsMT3", ("art", "infra")),
            ("wiki_mt4", "WikiTopicsMT4", ("sci", "health")),
        )
        for version in versions
    ),
    _inductive(
        "metafam",
        "Metafam",
        "single",
        "inductive_entity_relation_mtdea",
        target_validation=False,
    ),
    _inductive(
        "fbnell",
        "FBNELL",
        "single",
        "inductive_entity_relation_mtdea",
        target_validation=False,
    ),
)


ULTRA57 = (*TRANSDUCTIVE, *GRAIL, *ILPC, *HM, *INGRAM, *MTDEA)
BY_SLUG = {dataset.slug: dataset for dataset in ULTRA57}


if len(ULTRA57) != 57 or len(BY_SLUG) != 57:
    raise AssertionError("ULTRA-57 registry must contain 57 unique datasets")

