# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import logging
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from core.index_builder.faiss import cagra_upper_level_projection as projection
from core.index_builder.faiss.cagra_upper_level_projection import (
    max_reachable_share,
    project_links,
    project_upper_level,
)


def random_case(rng, n, nb0, links_per_node, skewed, with_empty_slots):
    """A CAGRA-like level-0 graph (distinct targets per row, no self-links), a promoted set and an HNSW-like
    layer-1 graph (distinct targets per row, some empty slots)."""
    pool = rng.choice(n, max(nb0 + 2, n // 8), replace=False)
    table = np.empty((n, nb0), dtype=np.int32)
    for r in range(n):
        if (
            skewed and rng.random() < 0.9
        ):  # many links to few nodes: many low in-degree nodes
            table[r] = rng.choice(pool[pool != r], nb0, replace=False)
        else:
            c = rng.choice(n - 1, nb0, replace=False)
            table[r] = c + (c >= r)
    if with_empty_slots:
        for r in np.flatnonzero(rng.random(n) < 0.3):
            table[r, nb0 - int(rng.integers(1, nb0 // 2 + 1)) :] = -1
    npro = int(rng.integers(links_per_node + 2, n // 3))
    promoted = np.sort(rng.choice(n, npro, replace=False)).astype(np.int32)
    layer1 = np.empty((npro, links_per_node), dtype=np.int32)
    for r in range(npro):
        c = rng.choice(npro - 1, links_per_node, replace=False)
        layer1[r] = c + (c >= r)
    layer1[rng.random(layer1.shape) < 0.15] = -1
    return table, promoted, layer1


def run_projection(table, promoted, layer1, max_share, protect):
    """project_links on a copy of table (contiguous rows). Returns (new table, links written)."""
    n, nb0 = table.shape
    neighbors = table.ravel().copy()
    offsets = np.arange(n + 1, dtype=np.int64) * nb0
    written = project_links(
        neighbors, offsets, nb0, promoted, layer1, max_share, protect
    )
    return neighbors.reshape(n, nb0), written


def in_degree(table):
    return np.bincount(table[table >= 0], minlength=table.shape[0])


def assert_invariants(before, after, promoted, max_share, protect):
    nb0 = before.shape[1]
    changed = np.flatnonzero((after != before).any(1))
    assert np.isin(changed, promoted).all(), "only promoted rows change"
    for r in changed:
        row = after[r]
        targets = row[row >= 0]
        assert len(np.unique(targets)) == len(targets), "no duplicate links"
        assert not (targets == r).any(), "no self-links"
        assert not ((before[r] >= 0) & (row < 0)).any(), "no slot is emptied"
        if not (before[r] < 0).any():
            assert (row != before[r]).sum() <= int(np.float32(max_share) * nb0), "cap"
    if protect:
        lost = (in_degree(before) > 0) & (in_degree(after) == 0)
        assert not lost.any(), "no node loses its last in-edge"


class TestProjectLinks:

    @pytest.mark.parametrize("seed", range(70))
    def test_invariants_on_random_graphs(self, seed):
        """Only promoted rows change, no duplicates or self-links, cap, no node loses its last in-edge; every
        layer-1 link either ends up in its row or its row is out of room"""
        rng = np.random.default_rng(seed)
        n = int(rng.integers(80, 1500))
        nb0 = int(rng.choice([8, 16, 32]))
        max_share = float(rng.choice([0.0, 0.25, 0.5, 1.0]))
        protect = bool(seed % 4 != 0)
        table, promoted, layer1 = random_case(
            rng,
            n,
            nb0,
            int(rng.choice([4, 8])),
            skewed=seed % 2 == 0,
            with_empty_slots=seed >= 40,
        )

        result, written = run_projection(table, promoted, layer1, max_share, protect)

        assert_invariants(table, result, promoted, max_share, protect)
        assert written == int((result != table).sum())
        for p, row in enumerate(layer1):
            for q in row[(row >= 0) & (row != p)]:
                a, b = promoted[p], promoted[q]
                for r, t in ((a, b), (b, a)):
                    if t in result[r]:
                        continue
                    # missing: its row used every empty slot first, then ran out of room
                    # (cap, or every remaining slot kept)
                    assert (result[r] >= 0).all()

    def test_connects_disconnected_clusters(self):
        """Two clusters with no link between them: a layer-1 link between them connects the graph both ways"""
        nb0, size = 4, 10
        n = 2 * size
        table = np.empty((n, nb0), dtype=np.int32)
        for c in range(2):
            for i in range(size):
                table[c * size + i] = [
                    c * size + (i + k) % size for k in range(1, nb0 + 1)
                ]
        promoted = np.array([3, 14], dtype=np.int32)
        layer1 = np.array([[1, -1], [0, -1]], dtype=np.int32)

        result, written = run_projection(table, promoted, layer1, 0.5, True)

        assert 14 in result[3] and 3 in result[14]
        assert written == 2  # the mutual link's second copy is already present
        seen = {0}
        queue = deque([0])
        while queue:
            for t in result[queue.popleft()]:
                if t not in seen:
                    seen.add(int(t))
                    queue.append(int(t))
        assert len(seen) == n

    def test_last_in_edge_is_protected(self):
        """The tail slot holding a node's only in-edge is skipped; the next slot is overwritten instead"""
        table = np.array(
            [
                [1, 2, 3, 4],  # node 0 (promoted); slot 3 -> 4 is node 4's only in-edge
                [2, 3, 0, 5],
                [3, 0, 1, 5],
                [0, 1, 2, 5],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            dtype=np.int32,
        )
        promoted = np.array([0, 5], dtype=np.int32)
        layer1 = np.array([[1], [-1]], dtype=np.int32)

        protected, _ = run_projection(table, promoted, layer1, 0.5, True)
        assert protected[0].tolist() == [1, 2, 5, 4]
        assert in_degree(protected)[4] == 1

        unprotected, _ = run_projection(table, promoted, layer1, 0.5, False)
        assert unprotected[0].tolist() == [1, 2, 3, 5]
        assert in_degree(unprotected)[4] == 0

    def test_zero_max_share_leaves_full_rows_unchanged(self):
        rng = np.random.default_rng(3)
        table, promoted, layer1 = random_case(
            rng, 400, 16, 8, skewed=True, with_empty_slots=False
        )

        result, written = run_projection(table, promoted, layer1, 0.0, True)

        np.testing.assert_array_equal(result, table)
        assert written == 0

    def test_only_listed_slots_are_written(self):
        """The neighbour table may hold upper-level slots between rows: only promoted rows' level-0 slots change"""
        rng = np.random.default_rng(5)
        table, promoted, layer1 = random_case(
            rng, 300, 8, 4, skewed=False, with_empty_slots=False
        )
        n, nb0 = table.shape
        # each node's level-0 row followed by 3 (empty) slots of another level
        stride = nb0 + 3
        neighbors = np.full(n * stride, -1, dtype=np.int32)
        offsets = np.arange(n + 1, dtype=np.int64) * stride
        for i in range(n):
            neighbors[offsets[i] : offsets[i] + nb0] = table[i]
        expected, _ = run_projection(table, promoted, layer1, 0.5, True)

        project_links(neighbors, offsets, nb0, promoted, layer1, 0.5, True)

        rows = neighbors.reshape(n, stride)
        np.testing.assert_array_equal(rows[:, :nb0], expected)
        assert (rows[:, nb0:] == -1).all()


class TestProjectUpperLevel:

    @staticmethod
    def _cpu_index(table, base_level_only=True, metric_type=1):
        n, nb0 = table.shape
        hnsw = SimpleNamespace(
            neighbors=table.ravel().copy(),
            offsets=np.arange(n + 1, dtype=np.int64) * nb0,
            nb_neighbors=lambda level: nb0,
        )
        return SimpleNamespace(
            hnsw=hnsw,
            ntotal=n,
            base_level_only=base_level_only,
            metric_type=metric_type,
        )

    def test_builds_layer1_from_promoted_vectors_and_projects(self):
        rng = np.random.default_rng(11)
        table, promoted, layer1 = random_case(
            rng, 200, 16, 8, skewed=True, with_empty_slots=False
        )
        n = table.shape[0]
        vectors = rng.random((n, 6)).astype(np.float16)
        cpu_index = self._cpu_index(table)

        with patch.object(
            projection, "_vec_view", side_effect=lambda v: v
        ), patch.object(
            projection, "draw_promoted", return_value=promoted
        ) as mock_draw, patch.object(
            projection, "build_layer1", return_value=layer1
        ) as mock_layer1:
            project_upper_level(
                cpu_index, vectors, m=4, ef_construction=64, skip_if_reachable=2.0
            )
        mock_draw.assert_called_once_with(n, 4)
        layer1_vectors, metric, links_per_node, ef_construction = mock_layer1.call_args[
            0
        ]
        assert layer1_vectors.dtype == np.float32
        assert layer1_vectors.flags.c_contiguous
        np.testing.assert_array_equal(
            layer1_vectors, vectors[promoted].astype(np.float32)
        )
        assert (metric, links_per_node, ef_construction) == (1, 4, 64)

        expected, written = run_projection(table, promoted, layer1, 0.5, True)
        assert written > 0
        np.testing.assert_array_equal(
            cpu_index.hnsw.neighbors.reshape(table.shape), expected
        )

    def test_fewer_than_two_promoted_is_a_no_op(self):
        table = np.array([[1], [0]], dtype=np.int32)
        cpu_index = self._cpu_index(table)

        with patch.object(
            projection, "_vec_view", side_effect=lambda v: v
        ), patch.object(
            projection, "draw_promoted", return_value=np.array([1], dtype=np.int32)
        ), patch.object(
            projection, "build_layer1"
        ) as mock_layer1:
            project_upper_level(
                cpu_index,
                np.zeros((2, 4), np.float32),
                m=16,
                ef_construction=100,
                skip_if_reachable=2.0,
            )
        mock_layer1.assert_not_called()
        np.testing.assert_array_equal(cpu_index.hnsw.neighbors, [1, 0])

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"base_level_only": False}, "base_level_only"),
            ({"m": 0}, "m > 0"),
            ({"max_share": 1.5}, "max_share"),
            ({"max_share": -0.1}, "max_share"),
            ({"num_vectors": 3}, "3 vectors for an index of 2"),
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs, message):
        table = np.array([[1], [0]], dtype=np.int32)
        cpu_index = self._cpu_index(table, kwargs.get("base_level_only", True))
        vectors = np.zeros((kwargs.get("num_vectors", 2), 4), np.float32)

        with pytest.raises(ValueError, match=message):
            project_upper_level(
                cpu_index,
                vectors,
                m=kwargs.get("m", 16),
                ef_construction=100,
                max_share=kwargs.get("max_share", 0.5),
            )

    @staticmethod
    def _two_rings(size=50, nb0=4):
        """Two clusters, each a ring: from any node only its own half (50%) is reachable"""
        table = np.empty((2 * size, nb0), dtype=np.int32)
        for c in range(2):
            for i in range(size):
                table[c * size + i] = [
                    c * size + (i + k) % size for k in range(1, nb0 + 1)
                ]
        return table

    def test_gate_skips_well_connected_graph(self):
        """Every node reachable: no layer-1 build, graph unchanged"""
        table = self._two_rings()
        table[0, -1] = 50  # ring 0 -> ring 1
        table[50, -1] = 0  # ring 1 -> ring 0
        cpu_index = self._cpu_index(table)
        vectors = np.zeros((table.shape[0], 4), np.float32)

        with patch.object(
            projection, "_vec_view", side_effect=lambda v: v
        ), patch.object(projection, "build_layer1") as mock_layer1:
            project_upper_level(cpu_index, vectors, m=2, ef_construction=16)
        mock_layer1.assert_not_called()
        np.testing.assert_array_equal(
            cpu_index.hnsw.neighbors.reshape(table.shape), table
        )

    def test_gate_projects_fragmented_graph(self):
        """Only half the graph reachable: the projection runs"""
        table = self._two_rings()
        promoted = np.array([3, 60], dtype=np.int32)
        cpu_index = self._cpu_index(table)
        vectors = np.zeros((table.shape[0], 4), np.float32)

        with patch.object(
            projection, "_vec_view", side_effect=lambda v: v
        ), patch.object(
            projection, "draw_promoted", return_value=promoted
        ), patch.object(
            projection,
            "build_layer1",
            return_value=np.array([[1], [0]], dtype=np.int32),
        ):
            project_upper_level(cpu_index, vectors, m=2, ef_construction=16)
        result = cpu_index.hnsw.neighbors.reshape(table.shape)
        assert 60 in result[3] and 3 in result[60]

    def test_max_reachable_share(self):
        table = self._two_rings()
        n, nb0 = table.shape
        offsets = np.arange(n + 1, dtype=np.int64) * nb0
        assert max_reachable_share(table.ravel(), offsets, nb0) == 0.5
        table[0, -1] = 50  # ring 0 reaches ring 1, not the other way
        assert max_reachable_share(table.ravel(), offsets, nb0, probes=n) == 1.0
        table[:, 2:] = -1  # empty slots are not followed
        assert max_reachable_share(table.ravel(), offsets, nb0, probes=n) == 0.5

    def test_failed_post_condition_leaves_graph_unchanged(self):
        """If the result would leave a node without an in-edge, nothing is written"""
        table = np.array(
            [
                [1, 2, 3, 4],  # node 0 (promoted); slot 3 -> 4 is node 4's only in-edge
                [2, 3, 0, 5],
                [3, 0, 1, 5],
                [0, 1, 2, 5],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            dtype=np.int32,
        )
        promoted = np.array([0, 5], dtype=np.int32)
        layer1 = np.array([[1], [-1]], dtype=np.int32)

        real_in_degree = projection._in_degree
        calls = []

        def in_degree_without_protection(neighbors, n, threads=8):
            """The second count (in-edges held by promoted rows) reports none: nothing gets protected"""
            calls.append(1)
            hist = real_in_degree(neighbors, n, threads)
            return hist if len(calls) == 1 else np.zeros_like(hist)

        neighbors = table.ravel().copy()
        offsets = np.arange(table.shape[0] + 1, dtype=np.int64) * 4
        with patch.object(projection, "_in_degree", in_degree_without_protection):
            with pytest.raises(RuntimeError, match="1 nodes without an in-edge"):
                project_links(neighbors, offsets, 4, promoted, layer1, 0.5, True)
        np.testing.assert_array_equal(neighbors.reshape(table.shape), table)

    @pytest.mark.parametrize(
        "error", [MemoryError("out of memory"), RuntimeError("boom")]
    )
    def test_failure_leaves_graph_unchanged(self, error, caplog):
        """Fail-safe: an error during the projection is logged, the graph is unchanged, and it returns False"""
        table = self._two_rings()
        cpu_index = self._cpu_index(table)
        vectors = np.zeros((table.shape[0], 4), np.float32)

        with patch.object(
            projection, "_vec_view", side_effect=lambda v: v
        ), patch.object(
            projection, "draw_promoted", return_value=np.array([3, 60], dtype=np.int32)
        ), patch.object(
            projection, "build_layer1", side_effect=error
        ):
            with caplog.at_level(logging.WARNING):
                project_upper_level(cpu_index, vectors, m=2, ef_construction=16)
        assert (
            "Upper-level projection failed, the base graph is unchanged" in caplog.text
        )
        np.testing.assert_array_equal(
            cpu_index.hnsw.neighbors.reshape(table.shape), table
        )
