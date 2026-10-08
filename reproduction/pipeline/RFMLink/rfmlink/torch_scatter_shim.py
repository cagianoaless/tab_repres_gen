"""Pure-PyTorch replacement for the compiled torch_scatter package.

torch_scatter has no prebuilt wheel for torch 2.10+cu128 (the PyG wheel index
stops at torch 2.9.1) and the system nvcc (CUDA 10.1) cannot compile it from
source against the tfms torch build. ULTRA imports torch_scatter directly, so
this module is registered under the name ``torch_scatter`` in ``sys.modules``
BEFORE importing ``ultra``. All functions delegate to
``torch_geometric.utils.scatter``, which uses native ``torch.scatter_reduce``.

This substitution is documented in results/audit/ultra_audit.json. It is not a
silent protocol change: numerical behaviour of scatter add/mean/max/min via
scatter_reduce matches the compiled kernels (up to floating-point reduction
order, which also varies inside torch_scatter itself).
"""

import sys
import types

import torch
from torch_geometric.utils import scatter as _pyg_scatter


def scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
    if out is not None:
        raise NotImplementedError("torch_scatter shim does not support out=")
    if reduce == "add":
        reduce = "sum"
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)


def scatter_add(src, index, dim=0, out=None, dim_size=None):
    return scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce="sum")


def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    return scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce="mean")


def scatter_max(src, index, dim=0, out=None, dim_size=None):
    """torch_scatter.scatter_max returns (values, argmax); ULTRA uses both."""
    if out is not None:
        raise NotImplementedError("torch_scatter shim does not support out=")
    if dim < 0:
        dim = src.dim() + dim
    size = src.size()
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    shape = list(size)
    shape[dim] = dim_size

    index_expanded = index
    if index.dim() == 1 and src.dim() > 1:
        view = [1] * src.dim()
        view[dim] = -1
        index_expanded = index.view(view).expand_as(src)

    values = torch.full(
        shape, float("-inf"), dtype=src.dtype, device=src.device
    ).scatter_reduce(dim, index_expanded, src, reduce="amax", include_self=True)
    match = values.gather(dim, index_expanded) == src
    positions = torch.arange(size[dim], device=src.device)
    view = [1] * src.dim()
    view[dim] = -1
    positions = positions.view(view).expand_as(src)
    argmax = torch.full(shape, -1, dtype=torch.long, device=src.device)
    argmax = argmax.scatter_reduce(
        dim,
        index_expanded,
        torch.where(match, positions, torch.full_like(positions, -1)),
        reduce="amax",
        include_self=True,
    )
    return values, argmax


def scatter_min(src, index, dim=0, out=None, dim_size=None):
    values, argmin = scatter_max(-src, index, dim=dim, out=out, dim_size=dim_size)
    return -values, argmin


def generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul="mul"):
    """Pure-torch replacement for ULTRA's compiled rspmm kernel.

    Semantics verified against the compiled CPU kernel (results/audit/
    ultra_audit.json): for each edge (u -> v) with type r,
        output[u] <sum>= edge_weight * (relation[r] <mul> input[v])
    i.e. aggregation happens at ``node_in`` pulling from ``node_out`` — the
    REVERSE of the standard PyG message-passing direction. The released ULTRA
    checkpoints were trained with this kernel, so these semantics are the
    correct ones (a standard PyG fallback produces wrong scores; verified:
    layer max_abs_diff 1.7 vs 1.8e-7 for this implementation).

    Shapes: edge_index (2, E); edge_type (E,); edge_weight (E,);
    relation (R, F); input (N, F) -> output (N, F).
    """
    node_in, node_out = edge_index[0], edge_index[1]
    rel_e = relation.index_select(0, edge_type)
    inp_e = input.index_select(0, node_out)
    if mul == "mul":
        msg = rel_e * inp_e
    elif mul == "add":
        msg = rel_e + inp_e
    else:
        raise ValueError(f"unsupported mul {mul!r}")
    msg = msg * edge_weight.unsqueeze(-1)
    num_nodes = input.shape[0]
    index = node_in.unsqueeze(-1).expand_as(msg)
    if sum == "add":
        out = torch.zeros_like(input).scatter_reduce(
            0, index, msg, reduce="sum", include_self=True
        )
    elif sum == "max":
        out = torch.full_like(input, float("-inf")).scatter_reduce(
            0, index, msg, reduce="amax", include_self=True
        )
    elif sum == "min":
        out = torch.full_like(input, float("inf")).scatter_reduce(
            0, index, msg, reduce="amin", include_self=True
        )
    else:
        raise ValueError(f"unsupported sum {sum!r}")
    return out


def install_rspmm():
    """Register pure-torch rspmm as ultra.rspmm.rspmm BEFORE importing ultra.

    Importing the real ultra.rspmm.rspmm triggers a cpp_extension JIT build at
    module level, which cannot succeed here (system nvcc 10.1 vs torch cu128).
    """
    if "ultra.rspmm.rspmm" in sys.modules:
        return
    rspmm_module = types.ModuleType("ultra.rspmm.rspmm")
    rspmm_module.generalized_rspmm = generalized_rspmm
    rspmm_module.__shim__ = True
    package = types.ModuleType("ultra.rspmm")
    package.generalized_rspmm = generalized_rspmm
    package.rspmm = rspmm_module
    package.__shim__ = True
    sys.modules["ultra.rspmm"] = package
    sys.modules["ultra.rspmm.rspmm"] = rspmm_module


def install():
    """Register this module as ``torch_scatter`` if the real one is absent."""
    if "torch_scatter" in sys.modules:
        return sys.modules["torch_scatter"]
    module = types.ModuleType("torch_scatter")
    module.scatter = scatter
    module.scatter_add = scatter_add
    module.scatter_mean = scatter_mean
    module.scatter_max = scatter_max
    module.scatter_min = scatter_min
    module.__shim__ = True

    composite = types.ModuleType("torch_scatter.composite")

    def scatter_softmax(src, index, dim=-1):
        max_per_index = scatter(src, index, dim=dim, reduce="max")
        recentered = src - max_per_index.gather(
            dim, index.expand_as(src) if index.dim() != src.dim() else index
        )
        exp = recentered.exp()
        denom = scatter_add(exp, index, dim=dim)
        return exp / denom.gather(
            dim, index.expand_as(src) if index.dim() != src.dim() else index
        ).clamp(min=1e-16)

    def scatter_log_softmax(src, index, dim=-1):
        return scatter_softmax(src, index, dim=dim).clamp(min=1e-16).log()

    composite.scatter_softmax = scatter_softmax
    composite.scatter_log_softmax = scatter_log_softmax
    module.composite = composite

    sys.modules["torch_scatter"] = module
    sys.modules["torch_scatter.composite"] = composite
    return module
