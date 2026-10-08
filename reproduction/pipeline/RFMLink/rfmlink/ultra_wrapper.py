"""Wrapper around the official ULTRA repository (external/ULTRA).

Responsibilities:
  * import ULTRA with the torch_scatter shim installed;
  * force the pure-PyG message passing path (the fused rspmm JIT kernel cannot
    be compiled here: system nvcc is CUDA 10.1, torch is cu128) — this uses
    ULTRA's own fallback branch in GeneralizedRelationalConv.propagate and is
    mathematically the same computation, just slower;
  * build ULTRA Data objects DIRECTLY from the unified processed npy triples
    (data/processed/<dataset>/) so entity/relation ids match the rest of the
    pipeline exactly — ULTRA's own dataset classes are NOT used because they
    assign ids by order of appearance and do not persist the vocabulary;
  * score batches of (h, r, t) candidate triples and capture the
    relation-conditioned candidate representation z(h,r,t): the input of the
    final EntityNBFNet MLP, shape (…, 128) for the released checkpoints.

Leakage protocol:
  * The message-passing graph contains ONLY training triples (+ their inverse
    copies, as required by ULTRA's architecture).
  * For candidates that ARE training triples (context positives), the direct
    edge (h,*,t) and (t,*,h) can be removed per batch via
    ``remove_direct_edges=True`` (same mechanism ULTRA/NBFNet use during
    training: BaseNBFNet.remove_easy_edges). Removal is batch-level: all edges
    connecting any (h,t) pair present in the batch are dropped for the whole
    batch.
  * Valid/test candidates are never part of the train graph, so no removal is
    needed there.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from . import torch_scatter_shim
from .paths import ULTRA_ROOT

_ULTRA_IMPORTED = False


def import_ultra():
    """Import ultra with two documented shims installed FIRST:

    1. torch_scatter -> pure-torch shim (no wheel for torch 2.10+cu128);
    2. ultra.rspmm.rspmm -> pure-torch generalized_rspmm with semantics
       verified against the compiled CPU kernel (aggregation at node_in from
       node_out; see torch_scatter_shim.generalized_rspmm docstring and
       results/audit/ultra_audit.json). ULTRA's own fused code path
       (GeneralizedRelationalConv.message_and_aggregate) then runs unmodified.

    NOTE: an earlier approach that forced the standard PyG message+aggregate
    fallback produced WRONG scores (the kernel direction is reversed relative
    to PyG conventions); it was detected by the CoDEx-Small validation and
    removed.
    """
    global _ULTRA_IMPORTED
    torch_scatter_shim.install()
    torch_scatter_shim.install_rspmm()
    if str(ULTRA_ROOT) not in sys.path:
        sys.path.insert(0, str(ULTRA_ROOT))
    import ultra  # noqa: F401

    _ULTRA_IMPORTED = True
    return ultra


DEFAULT_MODEL_CFG = {
    "relation_model": {
        "class": "RelNBFNet",
        "input_dim": 64,
        "hidden_dims": [64, 64, 64, 64, 64, 64],
        "message_func": "distmult",
        "aggregate_func": "sum",
        "short_cut": True,
        "layer_norm": True,
    },
    "entity_model": {
        "class": "EntityNBFNet",
        "input_dim": 64,
        "hidden_dims": [64, 64, 64, 64, 64, 64],
        "message_func": "distmult",
        "aggregate_func": "sum",
        "short_cut": True,
        "layer_norm": True,
    },
}


def load_ultra_model(checkpoint_path, device="cuda"):
    import_ultra()
    from ultra.models import Ultra

    model = Ultra(
        rel_model_cfg=dict(DEFAULT_MODEL_CFG["relation_model"]),
        entity_model_cfg=dict(DEFAULT_MODEL_CFG["entity_model"]),
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model.to(torch.device(device))


def build_train_graph(train_triples, num_entities, num_relations, device="cuda"):
    """Build the ULTRA Data object from train triples ONLY.

    train_triples: int64 array [n, 3] as (h, r, t) — note ULTRA stores edges as
    (h, t) pairs with a separate edge_type vector, and requires inverse copies
    with relation ids shifted by num_relations.
    """
    import_ultra()
    from torch_geometric.data import Data
    from ultra.tasks import build_relation_graph

    triples = torch.as_tensor(np.asarray(train_triples, dtype=np.int64))
    heads, rels, tails = triples[:, 0], triples[:, 1], triples[:, 2]

    target_edge_index = torch.stack([heads, tails], dim=0)
    target_edge_type = rels

    edge_index = torch.cat([target_edge_index, target_edge_index.flip(0)], dim=1)
    edge_type = torch.cat([target_edge_type, target_edge_type + num_relations])

    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_entities,
        target_edge_index=target_edge_index,
        target_edge_type=target_edge_type,
        num_relations=num_relations * 2,
    )
    data = build_relation_graph(data)
    return data.to(torch.device(device))


class RepresentationHook:
    """Captures the input of entity_model.mlp — z(h,r,t) per candidate."""

    def __init__(self, model):
        self.captured = None
        self.handle = model.entity_model.mlp.register_forward_pre_hook(self)

    def __call__(self, module, inputs):
        self.captured = inputs[0].detach()

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def score_triples(
    model,
    data,
    triples,
    batch_size=4,
    capture_representations=True,
    remove_direct_edges=False,
    removal_triples=None,
):
    """Score (h, r, t) triples; optionally capture z(h,r,t).

    triples: int64 array [n, 3] in unified (h, r, t) order.
    Returns (scores [n], representations [n, feature_dim] or None).

    ULTRA batches have shape (bs, num_candidates, 3) in (h, t, r) order; we use
    num_candidates=1. remove_direct_edges applies BaseNBFNet.remove_easy_edges
    per batch, which removes the target triple itself (both its direct edge
    and its inverse copy) from the message-passing graph.

    removal_triples: REQUIRED when `triples` are in INVERSE orientation
    (relation id >= num_relations). remove_easy_edges builds its match
    patterns as [r, r + R] with no modulo (ultra/base_nbfnet.py), so passing
    an inverse triple only removes the inverse copy and LEAKS the direct edge
    (bug found by external review 2026-07-12; contaminated the balanced
    primary variant). Pass the direct-orientation triples here; they are used
    for edge removal while `triples` are used for scoring.
    """
    triples = np.asarray(triples, dtype=np.int64)
    num_direct = data.num_relations // 2
    if remove_direct_edges and removal_triples is None:
        if (triples[:, 1] >= num_direct).any():
            raise ValueError(
                "score_triples: inverse-orientation triples with "
                "remove_direct_edges=True require removal_triples in direct "
                "orientation (see docstring; leakage bug otherwise)"
            )
        removal_triples = triples
    if removal_triples is not None:
        removal_triples = np.asarray(removal_triples, dtype=np.int64)
        if remove_direct_edges and (removal_triples[:, 1] >= num_direct).any():
            raise ValueError("removal_triples must be in direct orientation")
    device = next(model.parameters()).device
    scores = []
    representations = [] if capture_representations else None
    hook = RepresentationHook(model) if capture_representations else None

    try:
        for start in range(0, len(triples), batch_size):
            chunk = triples[start : start + batch_size]
            batch = torch.as_tensor(chunk, device=device)
            # unified (h, r, t) -> ULTRA (h, t, r)
            batch = batch[:, [0, 2, 1]].unsqueeze(1)  # (bs, 1, 3)
            if remove_direct_edges:
                rem = torch.as_tensor(
                    removal_triples[start : start + batch_size], device=device
                )[:, [0, 2, 1]].unsqueeze(1)
                h_index, t_index, r_index = rem.unbind(-1)
                graph = model.entity_model.remove_easy_edges(
                    data, h_index, t_index, r_index
                )
            else:
                graph = data
            batch_scores = model(graph, batch)  # (bs, 1)
            scores.append(batch_scores.squeeze(1).float().cpu().numpy())
            if capture_representations:
                rep = hook.captured  # (bs, 1, feature_dim)
                representations.append(rep.squeeze(1).float().cpu().numpy())
    finally:
        if hook is not None:
            hook.remove()

    scores = np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
    if capture_representations:
        representations = (
            np.concatenate(representations)
            if representations
            else np.zeros((0, 0), dtype=np.float32)
        )
    return scores, representations


@torch.no_grad()
def score_all_tails(model, data, h_r_pairs, batch_size=8):
    """Score (h, r, ?) against ALL entities. Returns [n_pairs, num_nodes].

    Used for the ULTRA-alone filtered-ranking baseline (fast path: one forward
    per query scores every candidate tail simultaneously). Head queries
    (?, r, t) must be rewritten by the caller as (t, r + num_relations, ?)
    using inverse relations, exactly as ULTRA's own evaluation does.
    """
    h_r_pairs = np.asarray(h_r_pairs, dtype=np.int64)
    device = next(model.parameters()).device
    num_nodes = data.num_nodes
    all_scores = []
    for start in range(0, len(h_r_pairs), batch_size):
        chunk = torch.as_tensor(h_r_pairs[start : start + batch_size], device=device)
        bs = len(chunk)
        h_index = chunk[:, 0].unsqueeze(1).expand(bs, num_nodes)
        r_index = chunk[:, 1].unsqueeze(1).expand(bs, num_nodes)
        t_index = (
            torch.arange(num_nodes, device=device).unsqueeze(0).expand(bs, num_nodes)
        )
        batch = torch.stack([h_index, t_index, r_index], dim=-1)  # (bs, num_nodes, 3)
        scores = model(data, batch)  # (bs, num_nodes)
        all_scores.append(scores.float().cpu().numpy())
    return (
        np.concatenate(all_scores)
        if all_scores
        else np.zeros((0, num_nodes), dtype=np.float32)
    )


@torch.no_grad()
def representations_all_tails(model, data, h_r_pairs, batch_size=8):
    """Like score_all_tails but also returns z for every candidate tail.

    Returns (scores [n, num_nodes], reprs [n, num_nodes, feature_dim]).
    Memory: num_nodes * feature_dim * batch floats per step — fine for small KGs.
    """
    h_r_pairs = np.asarray(h_r_pairs, dtype=np.int64)
    device = next(model.parameters()).device
    num_nodes = data.num_nodes
    hook = RepresentationHook(model)
    all_scores, all_reprs = [], []
    try:
        for start in range(0, len(h_r_pairs), batch_size):
            chunk = torch.as_tensor(
                h_r_pairs[start : start + batch_size], device=device
            )
            bs = len(chunk)
            h_index = chunk[:, 0].unsqueeze(1).expand(bs, num_nodes)
            r_index = chunk[:, 1].unsqueeze(1).expand(bs, num_nodes)
            t_index = (
                torch.arange(num_nodes, device=device)
                .unsqueeze(0)
                .expand(bs, num_nodes)
            )
            batch = torch.stack([h_index, t_index, r_index], dim=-1)
            scores = model(data, batch)
            all_scores.append(scores.float().cpu().numpy())
            all_reprs.append(hook.captured.float().cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(all_scores), np.concatenate(all_reprs)
