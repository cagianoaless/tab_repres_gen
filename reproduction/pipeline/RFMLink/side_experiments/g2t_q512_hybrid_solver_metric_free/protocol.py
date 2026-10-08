#!/usr/bin/env python3
"""Frozen metric-free protocol for hybrid B2v2 feature materialization."""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

PROTOCOL = "g2t_q512_hybrid_solver_metric_free_v1"
PARENT_PROTOCOL = "ultra57_rfmlink_q512_v1"

SCREEN_DATASETS = (
    "codex_small",
    "nell_ind_v1",
    "nl_ingram_75",
    "ilpc2022_small",
    "wdsinger",
    "metafam",
)

CONFIRMATION_DATASETS = (
    "wn18rr_ind_v1",
    "fb15k237_ind_v1",
    "nell_ind_v2",
    "yago310",
    "conceptnet100k",
    "nell995",
    "hetionet",
    "aristov4",
    "ilpc2022_large",
    "nell23k",
    "codex_medium",
    "fb15k237",
    "fb15k237_20",
    "wk_ingram_100",
    "fb15k237_10",
    "hm_3k",
    "hm_indigo",
    "wn18rr_ind_v4",
    "wiki_mt4_sci",
    "wiki_mt4_health",
    "wiki_mt3_art",
    "wiki_mt2_sci",
    "wiki_mt1_tax",
    "wiki_mt1_health",
    "fb_ingram_50",
    "fb_ingram_25",
    "wk_ingram_25",
    "fb15k237_ind_v4",
    "fb_ingram_75",
    "wk_ingram_75",
    "fb15k237_ind_v3",
    "fbnell",
    "wn18rr_ind_v2",
    "nl_ingram_25",
    "nl_ingram_100",
    "fb15k237_ind_v2",
    "nl_ingram_0",
    "nl_ingram_50",
    "dbpedia100k",
    "codex_large",
    "wn18rr",
    "fb15k237_50",
    "hm_5k",
    "wk_ingram_50",
    "wn18rr_ind_v3",
    "wiki_mt3_infra",
    "wiki_mt2_org",
    "hm_1k",
    "nell_ind_v3",
    "nell_ind_v4",
    "fb_ingram_100",
)

PANEL = (*SCREEN_DATASETS, *CONFIRMATION_DATASETS)
SEEDS = (101, 102, 103)

FAMILY_ROOT = ROOT / "results" / PROTOCOL
FEATURE_ROOT = FAMILY_ROOT / "feature_assets"
AUDIT_ROOT = ROOT / "results" / "analysis" / PROTOCOL
PARENT_ASSETS = ROOT / "results" / PARENT_PROTOCOL / "immutable_assets"
SCREEN_FEATURES_PATH = (
    ROOT / "RFMLink" / "side_experiments" / "g2t_q512_eigengap_safe" / "features.py"
)
STABILITY_AUDIT_ROOT = (
    ROOT / "results" / "analysis"
    / "g2t_q512_metric_free_large_eigsh_stability_v1"
)

assert len(SCREEN_DATASETS) == 6
assert len(CONFIRMATION_DATASETS) == 51
assert len(PANEL) == 57
assert len(set(PANEL)) == 57
