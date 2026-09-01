from __future__ import annotations

import numpy as np

QUERY_TRANSFORMS = ("raw", "within_query_zscore", "within_query_percentile")


def _validate(values: np.ndarray, group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    groups = np.asarray(group_ids)
    if matrix.ndim != 2:
        raise ValueError("query-relative values must be a matrix")
    if groups.ndim != 1 or len(groups) != len(matrix):
        raise ValueError("query group IDs are not row aligned")
    if not np.isfinite(matrix).all():
        raise ValueError("query-relative values contain non-finite entries")
    return matrix, groups


def _percentile_block(block: np.ndarray) -> np.ndarray:
    rows, columns = block.shape
    output = np.zeros((rows, columns), dtype=np.float32)
    if rows <= 1:
        return output
    for column in range(columns):
        values = np.asarray(block[:, column])
        order = np.argsort(values, kind="mergesort")
        ordered = values[order]
        boundaries = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, rows]
        ranks = np.empty(rows, dtype=np.float64)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        if boundaries.size > 2:
            output[:, column] = (ranks / (rows - 1)).astype(np.float32)
    return output


def within_query_transform(
    values: np.ndarray,
    *,
    mode: str,
    group_ids: np.ndarray | None = None,
    group_size: int | None = None,
) -> np.ndarray:
    """Transform each candidate group independently without reading labels.

    Exactly one of ``group_ids`` and ``group_size`` is required. Zero-variance
    columns map to zero. Percentiles use midranks scaled to [0, 1].
    """
    if mode not in QUERY_TRANSFORMS:
        raise ValueError(f"unsupported query transform: {mode}")
    matrix = np.asarray(values, dtype=np.float32)
    if mode == "raw":
        return matrix
    if (group_ids is None) == (group_size is None):
        raise ValueError("provide exactly one query grouping representation")
    fixed_group_size = group_size
    if fixed_group_size is not None:
        if fixed_group_size <= 0 or len(matrix) % fixed_group_size:
            raise ValueError("fixed query group size does not divide the rows")
        groups = None
    else:
        groups = np.asarray(group_ids)
        matrix, groups = _validate(matrix, groups)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("query-relative values must be a finite matrix")
    output = np.empty_like(matrix, dtype=np.float32)
    if fixed_group_size is not None:
        row_groups = (
            np.arange(start, start + fixed_group_size)
            for start in range(0, len(matrix), fixed_group_size)
        )
    else:
        assert groups is not None
        _, first = np.unique(groups, return_index=True)
        ordered_groups = groups[np.sort(first)]
        row_groups = (np.flatnonzero(groups == group) for group in ordered_groups)
    for rows in row_groups:
        block = matrix[rows]
        if mode == "within_query_zscore":
            mean = block.mean(axis=0, dtype=np.float64)
            scale = block.std(axis=0, dtype=np.float64)
            nonzero = scale > 0.0
            transformed = np.zeros(block.shape, dtype=np.float32)
            transformed[:, nonzero] = (
                (block[:, nonzero] - mean[nonzero]) / scale[nonzero]
            ).astype(np.float32)
        else:
            transformed = _percentile_block(block)
        output[rows] = transformed
    return output
