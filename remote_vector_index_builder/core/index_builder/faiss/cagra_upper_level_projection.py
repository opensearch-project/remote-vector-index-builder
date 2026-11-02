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

Before any of this, a gate measures how much of the base graph search can reach from a few random entry points
(breadth-first search over the neighbour table). Well-connected graphs are left as they are, so they do not pay
for the layer-1 build. The graph is only written once the whole projection has been computed and checked (no
node newly left without an in-edge); on any error it is left unchanged.

The index stays base-level only; the file format is unchanged. It needs only stock faiss: the level-0 graph is
read and written in place through zero-copy numpy views of ``hnsw.neighbors`` / ``hnsw.offsets``.

How the projection is computed
------------------------------
One vectorized pass over all promoted rows, no iteration. Links whose target is already in the row are
dropped, and that slot is kept. A node that has an in-edge from a row that is not promoted can never lose its
last in-edge (only promoted rows change); for the other nodes every slot holding them is kept. Each row then
writes its links into its empty slots and its overwritable slots from the end. When a row has more links than
room, its own layer-1 neighbours win first (nearest first), then the nodes that list it highest.
"""

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


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
    skip_if_reachable: float = 0.99,
) -> bool:
    """
    Builds an HNSW layer 1 over a promoted subset of ``vectors`` and projects its links into the base level of
    ``cpu_index`` in place, unless the base graph is already well connected.

    Args:
        cpu_index: base-level-only faiss.IndexHNSWCagra, from the plain GpuIndexCagra.copyTo
        vectors: the host array the GPU index was built from ([n, d], float32, float16 or int8), in id order
        m: HNSW m of the index; layer 1 has 2 * m links per node and promotion probability 1 / (2 * m)
        ef_construction: efConstruction of the layer-1 build
        max_share: max share of a full base-level row that projected links may overwrite (0..1)
        protect_last_in_edge: never overwrite a slot holding its target's last base-level in-edge
        skip_if_reachable: skip the projection when at least this share of the nodes is reachable from one of a
            few random entry points (> 1: always project)

    Returns:
        True if the projection ran, False if it was skipped. On an exception the graph is unchanged.
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

    hnsw = cpu_index.hnsw
    neighbors, offsets = _vec_view(hnsw.neighbors), _vec_view(hnsw.offsets)
    nb0 = int(hnsw.nb_neighbors(0))

    t0 = time.perf_counter()
    reachable = (
        max_reachable_share(neighbors, offsets, nb0, stop_at=skip_if_reachable)
        if skip_if_reachable <= 1.0
        else float("nan")
    )
    t_gate = time.perf_counter() - t0
    if reachable >= skip_if_reachable:
        logger.debug(
            f"Upper-level projection skipped: {reachable:.2%} of {n} nodes reachable "
            f"(gate_ms={t_gate * 1000:.1f})"
        )
        return False

    t0 = time.perf_counter()
    links_per_node = 2 * m
    promoted = draw_promoted(n, links_per_node)
    if len(promoted) < 2:
        logger.debug(f"Upper-level projection: {len(promoted)} promoted, nothing to do")
        return False

    promoted_vectors = np.ascontiguousarray(vectors[promoted], dtype=np.float32)
    layer1 = build_layer1(
        promoted_vectors, cpu_index.metric_type, links_per_node, ef_construction
    )
    del promoted_vectors
    t1 = time.perf_counter()

    added = project_links(
        neighbors,
        offsets,
        nb0,
        promoted,
        layer1,
        max_share,
        protect_last_in_edge,
    )
    t2 = time.perf_counter()
    logger.debug(
        f"Upper-level projection: {reachable:.2%} of {n} nodes reachable, "
        f"promoted={len(promoted)} links_added={added} gate_ms={t_gate * 1000:.1f} "
        f"layer1_build_ms={(t1 - t0) * 1000:.1f} projection_ms={(t2 - t1) * 1000:.1f}"
    )
    return True


# --------------------------------------------------------------------------------------------------------------
# Gate: reachability of the base graph
# --------------------------------------------------------------------------------------------------------------


def max_reachable_share(
    neighbors: np.ndarray,
    offsets: np.ndarray,
    nb0: int,
    stop_at: float = 1.0,
    probes: int = 3,
    seed: int = 12345,
) -> float:
    """
    Largest share of the nodes reachable (following links) from one of ``probes`` random entry points, which is
    what a base-level-only search sees. Stops early once a probe reaches ``stop_at``.

    One breadth-first search costs at most one pass over the neighbour table; on a fragmented graph each probe
    only visits its own component.
    """
    n = len(offsets) - 1
    best = 0.0
    for start in np.random.default_rng(seed).choice(n, min(probes, n), replace=False):
        best = max(best, _reachable_count(neighbors, offsets, nb0, int(start)) / n)
        if best >= stop_at:
            break
    return best


def _reachable_count(
    neighbors: np.ndarray, offsets: np.ndarray, nb0: int, start: int
) -> int:
    n = len(offsets) - 1
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    frontier = np.array([start], dtype=np.int64)
    slots = np.arange(nb0, dtype=np.int64)
    count = 1
    while frontier.size:
        nxt = neighbors[(offsets[frontier].astype(np.int64)[:, None] + slots).ravel()]
        nxt = nxt[nxt >= 0]
        nxt = np.unique(nxt[~visited[nxt]])
        visited[nxt] = True
        count += nxt.size
        frontier = nxt
    return count


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
) -> int:
    """
    Projects the layer-1 graph into the level-0 graph in place, in one vectorized pass.

    Args:
        neighbors: level-0 neighbour table (int32, -1 = empty); the row of node i is
            neighbors[offsets[i] : offsets[i] + nb0]. Modified in place.
        offsets: per-node offset of its row in ``neighbors`` (n + 1 entries)
        nb0: level-0 slots per node
        promoted: sorted ids of the layer-1 nodes
        layer1: layer-1 graph [len(promoted), links_per_node], positions into ``promoted``, -1 = empty
        max_share: max share of a full row that may be overwritten
        protect_last_in_edge: never overwrite a slot holding its target's last in-edge

    Returns:
        the number of links written into level 0
    """
    n = len(offsets) - 1
    npro = len(promoted)

    # 1. Every layer-1 link a-b gives row a the target b and row b the target a. Each link gets a priority for
    #    when its row runs out of room: the row's own layer-1 neighbours first, in HNSW's list order (nearest
    #    first), then the nodes that list the row, by how high they list it. Mutual links keep the better one.
    #    Keys row * npro + target come out sorted, so the links are grouped by row.
    k = layer1.shape[1]
    src = np.repeat(np.arange(npro, dtype=np.int64), k)
    dst = layer1.ravel().astype(np.int64)
    slot_in_list = np.tile(np.arange(k, dtype=np.int64), npro)
    valid = (dst >= 0) & (dst != src)
    src, dst, slot_in_list = src[valid], dst[valid], slot_in_list[valid]
    keyed = np.unique(
        np.concatenate(
            (
                (src * npro + dst) * (2 * k) + slot_in_list,  # own neighbour
                (dst * npro + src) * (2 * k) + k + slot_in_list,  # listed by src
            )
        )
    )
    first = np.r_[True, keyed[1:] // (2 * k) != keyed[:-1] // (2 * k)]
    keys, priority = keyed[first] // (2 * k), keyed[first] % (2 * k)
    del src, dst, slot_in_list, valid, keyed, first

    # 2. The level-0 rows of the promoted nodes.
    row_cols = (
        offsets[promoted].astype(np.int64)[:, None]
        + np.arange(nb0, dtype=np.int64)[None, :]
    )
    rows = neighbors[row_cols]  # [npro, nb0]
    rows_flat = rows.ravel()

    # 3. Links whose target is already in the row are dropped, and that slot is kept (layer 1 wants it too).
    #    Only promoted nodes are layer-1 targets, so only slots holding one are looked up.
    is_promoted = np.zeros(n, dtype=bool)
    is_promoted[promoted] = True
    slots = np.flatnonzero(_take_ids(is_promoted, rows_flat))
    slot_keys = (slots // nb0) * npro + np.searchsorted(promoted, rows_flat[slots])
    hit = _sorted_lookup(keys, slot_keys)
    found = hit >= 0
    keep = np.zeros(npro * nb0, dtype=bool)  # slots that must not be overwritten
    keep[slots[found]] = True
    present = np.zeros(keys.size, dtype=bool)
    present[hit[found]] = True
    keys, priority = keys[~present], priority[~present]
    link_row, target = keys // npro, promoted[keys % npro]
    by_priority = np.argsort(link_row * (2 * k) + priority, kind="stable")
    link_row, target = link_row[by_priority], target[by_priority]
    del keys, priority, by_priority, present, hit, found, slots, slot_keys

    # 4. Last in-edge protection, conservative so that no iteration is needed: only promoted rows lose slots,
    #    so a node keeps an in-edge for sure if one comes from a row that is not promoted. For the other nodes
    #    every slot holding them is kept.
    in_degree = None
    if protect_last_in_edge:
        in_degree = _in_degree(neighbors, n)
        only_in_promoted_rows = in_degree == _in_degree(rows_flat, n, 1)
        keep |= _take_ids(only_in_promoted_rows, rows_flat)

    # 5. Each row takes its links in order: empty slots first, then the unprotected slots from the end of the
    #    row (CAGRA's lowest-ranked links), at most max_share of a full row. Rows with empty slots are not capped.
    empty = rows < 0
    keep = keep.reshape(npro, nb0) & ~empty
    num_empty = empty.sum(1)
    overwritable = nb0 - num_empty - keep.sum(1)
    cap = np.where(num_empty > 0, nb0, int(np.float32(max_share) * nb0))
    budget = num_empty + np.minimum(cap, overwritable)

    row_start = np.searchsorted(link_row, np.arange(npro + 1))
    rank = (
        np.arange(link_row.size) - row_start[link_row]
    )  # position of the link within its row
    taken = rank < budget[link_row]
    link_row, target, rank = link_row[taken], target[taken], rank[taken]

    # the rank-th usable slot: rows without empty or kept slots use the tail directly
    slot = nb0 - 1 - rank
    special = np.flatnonzero(empty.any(1) | keep.any(1))
    if special.size:
        # usable slots in order: empty slots (ascending), then overwritable slots from the end
        order_key = np.where(empty[special], np.arange(nb0), 2 * nb0 - np.arange(nb0))
        order_key[keep[special]] = 3 * nb0
        slot_order = np.argsort(order_key, axis=1, kind="stable")
        position = np.full(npro, -1, dtype=np.int64)
        position[special] = np.arange(special.size)
        in_special = position[link_row] >= 0
        slot[in_special] = slot_order[position[link_row[in_special]], rank[in_special]]

    # 6. Post-condition, then write.
    flat = link_row * nb0 + slot
    if in_degree is not None:
        removed = rows_flat[flat]
        removed = removed[removed >= 0]
        delta = np.bincount(target, minlength=n) - np.bincount(removed, minlength=n)
        lost = (in_degree > 0) & (in_degree + delta <= 0)
        if lost.any():
            raise RuntimeError(
                f"upper-level projection would leave {int(lost.sum())} nodes without an in-edge"
            )
    neighbors[row_cols.ravel()[flat]] = target
    return int(target.size)


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
