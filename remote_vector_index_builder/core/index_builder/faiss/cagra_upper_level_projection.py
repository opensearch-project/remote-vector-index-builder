# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""
Projection of HNSW layer-1 links into the base layer of a CAGRA-built IndexHNSWCagra.

On datasets made of well-separated clusters the k-NN graph CAGRA prunes from is itself disconnected, so the
CAGRA graph splits into components and base-level-only search (random entry points) misses most of them. After
the plain ``GpuIndexCagra.copyTo`` this module:

1. promotes nodes with faiss's own HNSW level generator, as for an HNSW with ``2 * m`` links per node
   (probability ``1 / (2 * m)``, faiss rng seed 12345);
2. builds a temporary single-level ``IndexHNSWFlat`` (``2 * m`` links per node) over the promoted vectors.
   HNSW's incremental insertion creates near-neighbour links between clusters, which a pruned k-NN graph lacks;
3. projects every layer-1 link a-b into level 0 as a->b and b->a: empty slots first, then the tail of full rows
   (CAGRA's lowest-ranked links), at most ``max_share`` of a full row, never removing a node's last in-edge.
   Only the promoted nodes' rows change.

The index stays base-level only; the file format is unchanged. It needs only stock faiss: the level-0 graph is
read and written in place through zero-copy numpy views of ``hnsw.neighbors`` / ``hnsw.offsets``.

How the projection is computed
------------------------------
The result is defined by a sequential pass: promoted rows are processed one by one in id order, and each row
handles its links in order. A link is skipped if its target is already in the row; otherwise it fills the
row's first empty slot, or, while the row is below its cap, overwrites the next tail slot (walking from the
end) whose target keeps another in-edge.

Done literally that is a Python loop over millions of links, so the pass is vectorized over all rows at once.
Two things couple a link to others and cannot be known up front:

* *present*: a target that is in the row can be overwritten by an earlier link of the same row, after which
  the link adds it back. Mutual layer-1 links give a row the same target twice (the second copy is present iff
  the first was added and is still there).
* *protected*: a slot is kept only if its target's in-degree, at the moment the row is processed, is <= 1.
  That depends on the slots earlier rows overwrote and the links they added.

Both are solved as a fixed point: assign all rows from a guess, recompute *present* and *protected* from the
assignment, and redo only the rows whose inputs changed (typically 4-5 rounds, each over far fewer rows than
the last). A repair pass guarantees termination with no node losing its last in-edge.
"""

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# what happened to a projected link
_NOT_ADDED, _INTO_EMPTY, _OVERWRITE = 0, 1, 2


# --------------------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------------------


def project_upper_level(
    cpu_index: Any,
    vectors: np.ndarray,
    m: int,
    ef_construction: int,
    max_share: float = 0.5,
    protect_last_in_edge: bool = True,
) -> None:
    """
    Builds an HNSW layer 1 over a promoted subset of ``vectors`` and projects its links into the base level of
    ``cpu_index`` in place.

    Args:
        cpu_index: base-level-only faiss.IndexHNSWCagra, from the plain GpuIndexCagra.copyTo
        vectors: the host array the GPU index was built from ([n, d], float32, float16 or int8), in id order
        m: HNSW m of the index; layer 1 has 2 * m links per node and promotion probability 1 / (2 * m)
        ef_construction: efConstruction of the layer-1 build
        max_share: max share of a full base-level row that projected links may overwrite (0..1)
        protect_last_in_edge: never overwrite a slot holding its target's last base-level in-edge
    """
    if not cpu_index.base_level_only:
        raise ValueError("upper-level projection requires base_level_only = True")
    if m <= 0:
        raise ValueError(f"upper-level projection needs m > 0, got {m}")
    if not 0.0 <= max_share <= 1.0:
        raise ValueError(
            f"upper-level projection needs 0 <= max_share <= 1, got {max_share}"
        )
    n = int(cpu_index.ntotal)
    if vectors.shape[0] != n:
        raise ValueError(
            f"upper-level projection got {vectors.shape[0]} vectors for an index of {n}"
        )

    t0 = time.perf_counter()
    links_per_node = 2 * m
    promoted = draw_promoted(n, links_per_node)
    if len(promoted) < 2:
        logger.debug(f"Upper-level projection: {len(promoted)} promoted, nothing to do")
        return

    promoted_vectors = np.ascontiguousarray(vectors[promoted], dtype=np.float32)
    layer1 = build_layer1(
        promoted_vectors, cpu_index.metric_type, links_per_node, ef_construction
    )
    del promoted_vectors
    t1 = time.perf_counter()

    hnsw = cpu_index.hnsw
    added = project_links(
        _vec_view(hnsw.neighbors),
        _vec_view(hnsw.offsets),
        int(hnsw.nb_neighbors(0)),
        promoted,
        layer1,
        max_share,
        protect_last_in_edge,
    )
    t2 = time.perf_counter()
    logger.debug(
        f"Upper-level projection: promoted={len(promoted)} links_added={added} "
        f"layer1_build_ms={(t1 - t0) * 1000:.1f} projection_ms={(t2 - t1) * 1000:.1f}"
    )


# --------------------------------------------------------------------------------------------------------------
# Layer 1 (faiss)
# --------------------------------------------------------------------------------------------------------------


def _vec_view(vector: Any) -> np.ndarray:
    """Zero-copy numpy view of a faiss std::vector / MaybeOwnedVector: writes land in the C++ memory."""
    return faiss.rev_swig_ptr(vector.data(), vector.size())


def draw_promoted(n: int, links_per_node: int) -> np.ndarray:
    """
    Ids promoted to layer 1, drawn like faiss's HNSW does for an index with ``links_per_node`` links per node:
    faiss.HNSW rng (seed 12345), random_level() for ids 0..n-1 in order, level >= 1 is promoted.

    HNSW.prepare_level_tab runs the draw in C++; the per-level neighbour counts are zeroed first so it does not
    allocate neighbour tables.
    """
    draw = faiss.HNSW(links_per_node)
    draw.assign_probas.clear()
    draw.cum_nneighbor_per_level.clear()
    draw.set_default_probas(links_per_node, 1.0 / math.log(links_per_node))
    _vec_view(draw.cum_nneighbor_per_level)[:] = 0
    draw.prepare_level_tab(n, False)  # levels[i] = random_level() + 1
    levels = _vec_view(draw.levels)
    promoted = np.flatnonzero(levels >= 2).astype(np.int32 if n < 2**31 else np.int64)
    del levels, draw
    return promoted


def build_layer1(
    vectors: np.ndarray, metric: int, links_per_node: int, ef_construction: int
) -> np.ndarray:
    """
    Single-level IndexHNSWFlat over ``vectors`` (fp32, [np, d]). Returns its graph [np, links_per_node] (int32
    positions into ``vectors``, -1 = empty slot); the flat storage and the HNSW are freed before returning.
    """
    npro, d = vectors.shape
    index = faiss.IndexHNSWFlat(d, links_per_node, metric)
    hnsw = index.hnsw
    hnsw.assign_probas.clear()
    hnsw.cum_nneighbor_per_level.clear()
    hnsw.cum_nneighbor_per_level.push_back(0)
    hnsw.cum_nneighbor_per_level.push_back(links_per_node)
    faiss.copy_array_to_vector(
        np.ones(npro, dtype=np.int32), hnsw.levels
    )  # every node on level 0 only
    hnsw.efConstruction = ef_construction
    index.add(vectors)
    offsets = _vec_view(hnsw.offsets)
    if int(offsets[-1]) != npro * links_per_node:
        raise RuntimeError("layer-1 HNSW has non-uniform neighbour lists")
    graph = (
        _vec_view(hnsw.neighbors)[: npro * links_per_node]
        .reshape(npro, links_per_node)
        .copy()
    )
    del offsets, hnsw, index
    return graph


# --------------------------------------------------------------------------------------------------------------
# Projection (numpy)
# --------------------------------------------------------------------------------------------------------------


def project_links(
    neighbors: np.ndarray,
    offsets: np.ndarray,
    nb0: int,
    promoted: np.ndarray,
    layer1: np.ndarray,
    max_share: float,
    protect_last_in_edge: bool,
    max_iter: int = 64,
) -> int:
    """
    Projects the layer-1 graph into the level-0 graph in place (see the module docstring).

    Args:
        neighbors: level-0 neighbour table (int32, -1 = empty); the row of node i is
            neighbors[offsets[i] : offsets[i] + nb0]. Modified in place.
        offsets: per-node offset of its row in ``neighbors`` (n + 1 entries)
        nb0: level-0 slots per node
        promoted: sorted ids of the layer-1 nodes
        layer1: layer-1 graph [len(promoted), links_per_node], positions into ``promoted``, -1 = empty
        max_share: max share of a full row that may be overwritten
        protect_last_in_edge: never overwrite a slot holding its target's last in-edge
        max_iter: fixed-point iteration limit

    Returns:
        the number of links written into level 0
    """
    n = len(offsets) - 1
    links, lookup = _group_links_by_row(layer1, promoted)

    # level-0 rows of the promoted nodes: positions in `neighbors` and a copy of their contents
    row_cols = (
        offsets[promoted].astype(np.int64)[:, None]
        + np.arange(nb0, dtype=np.int64)[None, :]
    )
    rows = neighbors[row_cols]  # [npro, nb0]

    in_row, in_row_slot = _find_links_already_in_rows(
        links, lookup, layer1, rows, promoted, n
    )
    del lookup

    assignment = _RowAssignment(
        links,
        rows,
        in_row,
        in_row_slot,
        _in_degree(neighbors, n),
        promoted,
        max_share,
        protect_last_in_edge,
    )
    iterations = assignment.solve(max_iter)
    logger.debug(f"Upper-level projection converged in {iterations} iterations")
    return assignment.write_back(neighbors, row_cols)


class _Links(NamedTuple):
    """The projected links a->b (both directions of every layer-1 link), grouped by receiving row.

    Row r's links are ``row_start[r] : row_start[r + 1]``, in the order the sequential pass handles them. Rows
    are positions in ``promoted``.
    """

    row: np.ndarray  # [L] row receiving the link
    target_pos: np.ndarray  # [L] target, as a position in promoted
    target: np.ndarray  # [L] target node id
    row_start: np.ndarray  # [npro + 1]
    is_dup: np.ndarray  # [L] second copy of a mutual layer-1 link in the same row
    first_copy: np.ndarray  # [L] for a duplicate: index of its first copy, else -1


class _LinkLookup(NamedTuple):
    """Indexes into _Links used to find each row's link to a given target."""

    valid: (
        np.ndarray
    )  # [npro, k] layer-1 slot holds a link (not empty, not a self-link)
    own_start: np.ndarray  # [npro] index of the row's first own layer-1 link
    reverse_keys: np.ndarray  # [E] sorted row * npro + source of the reverse links
    reverse_link: np.ndarray  # [E] index in _Links of each reverse link, same order


def _group_links_by_row(
    layer1: np.ndarray, promoted: np.ndarray
) -> Tuple[_Links, _LinkLookup]:
    """
    Groups the 2E links of the layer-1 graph (E valid entries) by receiving row, in sequential order.

    Every valid layer-1 entry (i, k) -> j emits i->j and j->i, and the links are processed in that emission
    order. Grouped by row, row r therefore receives
        [reverse links from sources i < r, ascending i] + [its own layer-1 list] + [reverse links from i > r].
    Own lists are already in place, so only the reverse links are sorted (stable, by row); every link's
    position then follows from per-row counts.
    """
    npro = layer1.shape[0]
    pos = np.arange(npro, dtype=np.int32)
    valid = (layer1 >= 0) & (layer1 != pos[:, None])
    own_count = valid.sum(1)
    src = np.repeat(pos, own_count)  # entry e = valid (i, k), in emission order
    dst = layer1[valid]
    num_entries = dst.size

    by_row = _stable_argsort(dst, npro)  # reverse links j->i, grouped by row j
    reverse_row, reverse_src = dst[by_row], src[by_row]
    reverse_count = np.bincount(dst, minlength=npro)
    reverse_before = np.bincount(dst[src < dst], minlength=npro)  # from sources < row
    row_start = _segment_starts(own_count + reverse_count)
    own_start = row_start[:-1] + reverse_before

    e = np.arange(num_entries, dtype=np.int64)
    own_link = e + np.repeat(own_start - _segment_starts(own_count)[:-1], own_count)
    reverse_rank = e - np.repeat(_segment_starts(reverse_count)[:-1], reverse_count)
    reverse_link = (
        np.repeat(row_start[:-1], reverse_count)
        + reverse_rank
        + np.repeat(own_count, reverse_count)
        * (reverse_rank >= np.repeat(reverse_before, reverse_count))
    )
    del e, reverse_rank

    num_links = 2 * num_entries
    target_pos = np.empty(num_links, dtype=np.int32)
    target_pos[own_link] = dst
    target_pos[reverse_link] = reverse_src
    reverse_keys = reverse_row.astype(np.int64) * npro + reverse_src

    # A mutual pair (i -> j and j -> i both in layer 1) gives row i the target j twice: from its own list and
    # as a reverse link from j (and the same for row j). Entry (i, k) -> j with j < i is mutual iff row i has a
    # reverse link from j, i.e. key i * npro + j is among the reverse keys; the later copy is the duplicate.
    lower = np.flatnonzero(dst < src)
    keys = src[lower].astype(np.int64) * npro + dst[lower]
    hit = _sorted_lookup(reverse_keys, keys)
    found = hit >= 0
    entry_i, rev_i = (
        lower[found],
        hit[found],
    )  # (i -> j) and the reverse link j -> i in row i
    entry_j = by_row[rev_i]  # the partner entry (j -> i)
    rev_j = np.empty(
        num_entries, dtype=np.int64
    )  # reverse link of each entry, by entry
    rev_j[by_row] = np.arange(num_entries, dtype=np.int64)
    copy_a = np.concatenate((own_link[entry_i], own_link[entry_j]))
    copy_b = np.concatenate((reverse_link[rev_i], reverse_link[rev_j[entry_i]]))
    is_dup = np.zeros(num_links, dtype=bool)
    first_copy = np.full(num_links, -1, dtype=np.int64)
    later = np.maximum(copy_a, copy_b)
    is_dup[later] = True
    first_copy[later] = np.minimum(copy_a, copy_b)

    row = np.repeat(pos, own_count + reverse_count)
    links = _Links(row, target_pos, promoted[target_pos], row_start, is_dup, first_copy)
    return links, _LinkLookup(valid, own_start, reverse_keys, reverse_link)


def _find_links_already_in_rows(
    links: _Links,
    lookup: _LinkLookup,
    layer1: np.ndarray,
    rows: np.ndarray,
    promoted: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every link: is its target in the row before the projection, and at which slot.

    Only promoted nodes are layer-1 targets, so only the (row, slot) pairs holding a promoted node (about 2 per
    row) are looked up: in the row's own layer-1 list, and among the reverse links of the row.
    """
    npro, nb0 = rows.shape
    is_promoted = np.zeros(n, dtype=bool)
    is_promoted[promoted] = True
    slots = np.flatnonzero(_take_ids(is_promoted, rows.ravel()))
    row, slot = slots // nb0, slots % nb0
    target_pos = np.searchsorted(promoted, rows.ravel()[slots]).astype(np.int32)

    in_row = np.zeros(len(links.row), dtype=bool)
    in_row_slot = np.zeros(len(links.row), dtype=np.int16)

    # in the row's own layer-1 list: index = own_start + rank of the entry among the row's valid entries
    match = (layer1[row] == target_pos[:, None]) & lookup.valid[row]
    has = np.flatnonzero(match.any(1))
    k = match[has].argmax(1)
    rank = (
        lookup.valid[row[has]] & (np.arange(layer1.shape[1])[None, :] < k[:, None])
    ).sum(1)
    own = lookup.own_start[row[has]] + rank
    in_row[own] = True
    in_row_slot[own] = slot[has]

    # a reverse link from the target
    hit = _sorted_lookup(lookup.reverse_keys, row.astype(np.int64) * npro + target_pos)
    has = np.flatnonzero(hit >= 0)
    rev = lookup.reverse_link[hit[has]]
    in_row[rev] = True
    in_row_slot[rev] = slot[has]
    return in_row, in_row_slot


class _RowAssignment:
    """
    Assigns every link a slot (or none), vectorized over rows, and iterates to the sequential result.

    State per link: ``present`` (target in the row when the link is handled; initial guess: in the original
    row), ``status`` and ``slot``. State per slot of a promoted row: ``protected`` and ``overwritten_by`` (the
    link that overwrote it, or ``none``). Slots are indexed flat: row * nb0 + slot.
    """

    def __init__(
        self,
        links: _Links,
        rows: np.ndarray,
        in_row: np.ndarray,
        in_row_slot: np.ndarray,
        in_degree: np.ndarray,
        promoted: np.ndarray,
        max_share: float,
        protect_last_in_edge: bool,
    ):
        npro, nb0 = rows.shape
        num_links = len(links.row)
        self.links = links
        self.npro, self.nb0, self.n = npro, nb0, len(in_degree)
        self.rows = rows
        self.in_degree = in_degree
        self.protect = protect_last_in_edge

        self.link_count = np.diff(links.row_start)
        self.row_base = (
            links.row.astype(np.int64) * nb0
        )  # flat index of slot 0 of the link's row
        has_links = self.link_count > 0

        # Rows with empty slots (CAGRA never produces them) are filled first and are not capped.
        empty = rows < 0
        self.num_empty = empty.sum(1).astype(np.int32)
        not_full = self.num_empty > 0
        self.row_cap = np.where(not_full, nb0, int(np.float32(max_share) * nb0)).astype(
            np.int32
        )
        self.has_empty = bool((not_full & has_links).any())
        self.filled_rows = (
            rows.copy() if self.has_empty else rows
        )  # rows after the fill
        self.empty_order: Any = None
        self.filled: Any = None
        self.empty_slots: Any = None
        if self.has_empty:
            self.empty_order = np.argsort(~empty, axis=1, kind="stable")
            self.filled = np.zeros(npro * nb0, dtype=bool)
            self.empty_slots = np.flatnonzero(empty.ravel() & np.repeat(has_links, nb0))

        # present: from the original row, at slot in_row_slot (until overwritten); duplicates: guess added
        self.in_row = in_row
        self.in_row_slot = in_row_slot
        self.in_row_links = np.flatnonzero(in_row)
        self.in_row_flat = (
            self.row_base[self.in_row_links] + in_row_slot[self.in_row_links]
        )
        self.dups = np.flatnonzero(links.is_dup)
        self.present = in_row.copy()
        self.present[self.dups] = True

        self.status = np.zeros(num_links, dtype=np.int8)
        self.slot = np.zeros(num_links, dtype=np.int16)
        self.none = num_links
        self.overwritten_by: Any = np.full(
            npro * nb0,
            num_links,
            dtype=np.int64 if num_links >= 2**31 - 1 else np.int32,
        )
        self.protected = np.zeros((npro, nb0), dtype=bool)
        if protect_last_in_edge:
            self._init_protection(promoted, has_links)

    # ---- last in-edge protection --------------------------------------------------------------------------

    def _init_protection(self, promoted: np.ndarray, has_links: np.ndarray) -> None:
        """Only "weak" targets can be protected: in-degree minus the in-edges held by promoted rows <= 1 (only
        promoted rows lose slots). Tracks their slots in promoted rows and the links that add an in-edge.
        """
        nb0 = self.nb0
        rows_flat = self.rows.ravel()
        self.weak = (self.in_degree - _in_degree(rows_flat, self.n, 1)) <= 1
        weak_slots = np.flatnonzero(_take_ids(self.weak, rows_flat))
        self.weak_slots = weak_slots[has_links[weak_slots // nb0]]
        self.weak_targets = rows_flat[self.weak_slots].astype(np.int64)
        self.weak_rows = self.weak_slots // nb0
        self.weak_links = np.flatnonzero(self.weak[promoted][self.links.target_pos])
        # first guess: the in-degree before the projection
        self.protected.ravel()[self.weak_slots] = self.in_degree[self.weak_targets] <= 1

    def _in_edge_events(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """In-degree changes of weak targets under the current assignment: (target, row, +1 / -1)."""
        none, nb0 = self.none, self.nb0
        removed = self.overwritten_by[self.weak_slots] < none
        targets = [self.weak_targets[removed]]
        rows = [self.weak_rows[removed]]
        deltas = [-np.ones(int(removed.sum()), np.int64)]
        status = self.status[self.weak_links]
        added = self.weak_links[(status == _INTO_EMPTY) | (status == _OVERWRITE)]
        targets.append(self.links.target[added].astype(np.int64))
        rows.append(self.links.row[added].astype(np.int64))
        deltas.append(np.ones(added.size, np.int64))
        if self.has_empty:  # a filled empty slot overwritten later in the row
            slots = self.empty_slots[
                self.filled[self.empty_slots]
                & (self.overwritten_by[self.empty_slots] < none)
            ]
            filled_flat = self.filled_rows.ravel()
            slots = slots[self.weak[filled_flat[slots]]]
            targets.append(filled_flat[slots].astype(np.int64))
            rows.append(slots // nb0)
            deltas.append(-np.ones(slots.size, np.int64))
        return np.concatenate(targets), np.concatenate(rows), np.concatenate(deltas)

    def update_protection(self) -> np.ndarray:
        """Protects slot (r, s) with target t iff t's in-degree when row r is handled (before + changes made by
        rows < r) is <= 1. Returns the rows whose protection changed."""
        npro, nb0 = self.npro, self.nb0
        protected = self.protected.ravel()
        targets, rows, deltas = self._in_edge_events()
        keys = targets * (npro + 1) + rows
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        delta_prefix = np.concatenate(([0], np.cumsum(deltas[order])))

        slots, slot_targets, slot_rows = (
            self.weak_slots,
            self.weak_targets,
            self.weak_rows,
        )
        own_add = None
        if (
            self.has_empty
        ):  # filled empty slots holding a weak target are candidates too
            filled_flat = self.filled_rows.ravel()
            fill = self.empty_slots[self.filled[self.empty_slots]]
            fill = fill[self.weak[filled_flat[fill]]]
            protected[self.empty_slots] = False
            slots = np.concatenate((slots, fill))
            slot_targets = np.concatenate(
                (slot_targets, filled_flat[fill].astype(np.int64))
            )
            slot_rows = np.concatenate((slot_rows, fill // nb0))
            own_add = np.concatenate(
                (np.zeros(self.weak_slots.size, np.int64), np.ones(fill.size, np.int64))
            )

        # in-degree of the target when the row is handled: prefix sum of its events in rows < row
        first_event = np.searchsorted(keys, slot_targets * (npro + 1), "left")
        row_event = np.searchsorted(keys, slot_targets * (npro + 1) + slot_rows, "left")
        degree = (
            self.in_degree[slot_targets]
            + delta_prefix[row_event]
            - delta_prefix[first_event]
        )
        if own_add is not None:
            degree += own_add  # a filled slot's own add came earlier in its row
        new_protected = degree <= 1
        changed = slots[new_protected != protected[slots]] // nb0
        protected[slots] = new_protected
        return np.unique(changed)

    def repair(self) -> np.ndarray:
        """Safety net if the fixed point does not settle: a weak target that would lose its last in-edge keeps
        the in-edge a sequential pass reaches last (highest row, then lowest slot). Returns the rows to redo.
        """
        nb0 = self.nb0
        targets, _, deltas = self._in_edge_events()
        total = np.zeros(self.n, np.int64)
        np.add.at(total, targets, deltas)
        degree = self.in_degree[targets]
        lost = np.unique(targets[(degree > 0) & (degree + total[targets] <= 0)])
        if lost.size == 0:
            return lost
        is_lost = np.zeros(self.n, bool)
        is_lost[lost] = True
        filled_flat = self.filled_rows.ravel()
        slots = (
            np.concatenate((self.weak_slots, self.empty_slots))
            if self.has_empty
            else self.weak_slots
        )
        slots = slots[
            (self.overwritten_by[slots] < self.none) & (filled_flat[slots] >= 0)
        ]
        slots = slots[is_lost[filled_flat[slots]]]
        slot_targets = filled_flat[slots]
        visit_order = (slots // nb0) * nb0 + (nb0 - 1 - slots % nb0)
        order = np.lexsort((visit_order, slot_targets))
        last = order[np.r_[slot_targets[order][1:] != slot_targets[order][:-1], True]]
        self.protected.ravel()[slots[last]] = True
        return np.unique(slots[last] // nb0)

    # ---- assignment ---------------------------------------------------------------------------------------

    def assign(self, todo: np.ndarray, link_idx: Any) -> np.ndarray:
        """
        Assigns the links of rows ``todo`` (``link_idx``: their link indexes; None = every row) given the
        current ``present`` and ``protected``, then recomputes ``present``. Returns the rows whose ``present``
        changed.
        """
        links, nb0, npro, none = self.links, self.nb0, self.npro, self.none
        every = link_idx is None

        def take(a: np.ndarray) -> np.ndarray:
            """Per-link array restricted to the links handled."""
            return a if every else a[link_idx]

        def link(i: np.ndarray) -> np.ndarray:
            """Local link index -> global."""
            return i if every else link_idx[i]

        def put(a: np.ndarray, values: np.ndarray) -> None:
            if every:
                a[:] = values
            else:
                a[link_idx] = values

        lens = self.link_count[todo]
        link_row = take(links.row)
        was_present = take(self.present)
        new = ~was_present

        # rank of each new link among the new links of its row
        new_seen = np.cumsum(new, dtype=np.int64)
        rank = new_seen - new
        rank -= np.repeat(
            np.concatenate(([0], new_seen))[_segment_starts(lens)[:-1]], lens
        )

        # empty slots take the first new links of their row
        status = np.zeros(len(link_row), dtype=np.int8)
        if self.has_empty:
            num_empty = self.num_empty[link_row]
            into_empty = new & (rank < num_empty)
            candidate = new & ~into_empty
            tail_rank = rank - num_empty
            self.filled_rows[todo] = self.rows[todo]
            self.filled.reshape(npro, nb0)[todo] = False
            fill = np.flatnonzero(into_empty)
            fill_row = link_row[fill]
            fill_slot = self.empty_order[fill_row, rank[fill]]
            self.filled_rows[fill_row, fill_slot] = links.target[link(fill)]
            self.filled[fill_row.astype(np.int64) * nb0 + fill_slot] = True
            status[fill] = _INTO_EMPTY
            self.slot[link(fill)] = fill_slot
        else:
            candidate, tail_rank = new, rank

        # the next new links overwrite the row's tail, up to the cap and the unprotected slots
        protected = self.protected[todo]
        unprotected = (nb0 - protected.sum(1)).astype(np.int32)
        limit = np.minimum(self.row_cap[todo], unprotected)
        over = np.flatnonzero(candidate & (tail_rank < np.repeat(limit, lens)))
        over_slot = (
            nb0 - 1 - tail_rank[over]
        )  # no protected slot: the j-th slot from the end
        with_protected = np.flatnonzero(protected.any(1))
        if with_protected.size:  # else the j-th unprotected slot from the end
            tail_slots = (
                nb0
                - 1
                - np.argsort(protected[with_protected, ::-1], axis=1, kind="stable")
            )
            local_row = np.full(len(todo), -1, dtype=np.int64)
            local_row[with_protected] = np.arange(with_protected.size)
            r = local_row[np.repeat(np.arange(len(todo)), lens)[over]]
            sel = r >= 0
            over_slot[sel] = tail_slots[r[sel], tail_rank[over[sel]]]
        over_link = link(over)
        if every:
            self.overwritten_by.fill(none)
        else:
            self.overwritten_by.reshape(npro, nb0)[todo] = none
        self.overwritten_by[self.row_base[over_link] + over_slot] = over_link
        status[over] = _OVERWRITE
        self.slot[over_link] = over_slot
        put(self.status, status)

        # present = the target is still in the row when the link is handled. From the original row: its slot
        # was not overwritten by an earlier link (>=: a link is not displaced by itself).
        now_present = np.zeros(len(link_row), dtype=bool)
        if every:
            now_present[self.in_row_links] = (
                self.overwritten_by[self.in_row_flat] >= self.in_row_links
            )
        else:
            local = np.flatnonzero(self.in_row[link_idx])
            glob = link_idx[local]
            now_present[local] = (
                self.overwritten_by[self.row_base[glob] + self.in_row_slot[glob]]
                >= glob
            )
        # A duplicate: its first copy was added (a tail slot is never overwritten twice; a filled empty slot
        # can be overwritten by a later link).
        dup = self.dups if every else np.flatnonzero(links.is_dup[link_idx])
        if dup.size:
            dup_link = link(dup)
            first = links.first_copy[dup_link]
            first_status = self.status[first]
            now_present[dup] |= first_status == _OVERWRITE
            filled = np.flatnonzero(first_status == _INTO_EMPTY)
            if filled.size:
                f = first[filled]
                now_present[dup[filled]] |= (
                    self.overwritten_by[self.row_base[f] + self.slot[f]]
                    > dup_link[filled]
                )
        changed = link_row[
            now_present != was_present
        ]  # sorted: links are grouped by row
        put(self.present, now_present)
        return (
            changed[np.r_[True, changed[1:] != changed[:-1]]]
            if changed.size
            else changed
        )

    def _links_of(self, todo: np.ndarray) -> np.ndarray:
        lens = self.link_count[todo]
        return np.arange(int(lens.sum()), dtype=np.int64) + np.repeat(
            self.links.row_start[todo] - _segment_starts(lens)[:-1], lens
        )

    def solve(self, max_iter: int) -> int:
        """Iterates assign / update_protection over the rows that change. Returns the iteration count."""
        todo = np.flatnonzero(self.link_count > 0)
        link_idx = None
        repairing = False  # once repairs start, protection is frozen
        for iteration in range(1, max_iter + 1):
            changed = self.assign(todo, link_idx)
            if self.protect:
                if not repairing:
                    reprotected = self.update_protection()
                    if reprotected.size:
                        changed = np.union1d(changed, reprotected)
                    if changed.size == 0:
                        changed = self.repair()
                        repairing = changed.size > 0
                    elif iteration >= max_iter // 2:
                        repairing = True
                elif changed.size == 0:
                    changed = self.repair()
            if changed.size == 0:
                return iteration
            todo = changed
            link_idx = self._links_of(todo)
        raise RuntimeError(
            f"upper-level projection did not converge in {max_iter} iterations"
        )

    def write_back(self, neighbors: np.ndarray, row_cols: np.ndarray) -> int:
        """Writes the added links into ``neighbors`` (fills first: a filled slot may be overwritten later).
        Returns the number of links written."""
        written = np.flatnonzero(self.status == _OVERWRITE)
        if self.has_empty:
            written = np.concatenate(
                (np.flatnonzero(self.status == _INTO_EMPTY), written)
            )
        flat = self.row_base[written] + self.slot[written]
        neighbors[row_cols.ravel()[flat]] = self.links.target[written]
        return int(written.size)


# --------------------------------------------------------------------------------------------------------------
# numpy helpers
# --------------------------------------------------------------------------------------------------------------


def _in_degree(neighbors: np.ndarray, n: int, threads: int = 8) -> np.ndarray:
    """In-degree of nodes 0..n-1 in ``neighbors`` (int32, -1 ignored). faiss.ivec_hist releases the GIL, so
    chunks are counted in parallel; np.bincount would make an int64 copy of the whole graph.
    """
    threads = max(1, min(threads, neighbors.size // (1 << 22)))
    if threads == 1:
        hist = np.zeros(n, dtype=np.int32)
        faiss.ivec_hist(
            neighbors.size, faiss.swig_ptr(neighbors), n, faiss.swig_ptr(hist)
        )
        return hist
    bounds = np.linspace(0, neighbors.size, threads + 1).astype(np.int64)
    hists = np.zeros((threads, n), dtype=np.int32)

    def count(t: int) -> None:
        chunk = neighbors[bounds[t] : bounds[t + 1]]
        faiss.ivec_hist(chunk.size, faiss.swig_ptr(chunk), n, faiss.swig_ptr(hists[t]))

    with ThreadPoolExecutor(threads) as executor:
        list(executor.map(count, range(threads)))
    return hists.sum(axis=0, dtype=np.int32)


def _take_ids(flags: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """flags[ids] with False for empty slots (id -1)."""
    out = flags[ids]
    if np.count_nonzero(ids >= 0) != ids.size:
        out[ids < 0] = False
    return out


def _sorted_lookup(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Index of each key in ``sorted_keys``, or -1."""
    if sorted_keys.size == 0:
        return np.full(keys.size, -1, dtype=np.int64)
    idx = np.minimum(np.searchsorted(sorted_keys, keys), sorted_keys.size - 1)
    return np.where(sorted_keys[idx] == keys, idx, -1)


def _stable_argsort(keys: np.ndarray, kmax: int) -> np.ndarray:
    """Stable argsort of non-negative integer keys < kmax (numpy radix sort on 16-bit digits)."""
    if kmax <= 1 << 16:
        return np.argsort(keys.astype(np.uint16), kind="stable")
    if kmax <= 1 << 32:
        low = np.argsort((keys & 0xFFFF).astype(np.uint16), kind="stable")
        high = np.argsort((keys[low] >> 16).astype(np.uint16), kind="stable")
        return low[high]
    return np.argsort(keys, kind="stable")


def _segment_starts(counts: np.ndarray) -> np.ndarray:
    """[0, c0, c0 + c1, ...]: start of each segment given the segment lengths."""
    starts = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    return starts
