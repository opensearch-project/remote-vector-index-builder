# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from core.index_builder.faiss import cagra_upper_level_projection as projection
from core.index_builder.faiss.cagra_upper_level_projection import (
    project_links,
    project_upper_level,
)


def reference_projection(table, layer1, promoted, max_share, protect):
    """Line-by-line sequential version of the projection: rows processed one by one in id order.
    table: [n, nb0] level-0 graph, modified in place. Returns the number of links written.
    """
    n, nb0 = table.shape
    cap = int(np.float32(max_share) * nb0)
    indeg = np.bincount(table[table >= 0], minlength=n).astype(np.int64)
    ids = promoted.tolist()
    links = [[] for _ in ids]
    for i in range(len(ids)):
        for j in layer1[i].tolist():
            if j < 0 or j == i:
                continue
            links[i].append(ids[j])
            links[j].append(ids[i])
    written = 0
    for r, row_links in enumerate(links):
        if not row_links:
            continue
        row = table[ids[r]].tolist()
        row_cap = nb0 if -1 in row else cap
        replaced, tail = 0, nb0 - 1
        for b in row_links:
            if b in row:
                continue
            if -1 in row:
                row[row.index(-1)] = b
                indeg[b] += 1
                written += 1
                continue
            if replaced >= row_cap:
                continue
            while tail >= 0:
                t = row[tail]
                if not protect or indeg[t] > 1:
                    indeg[t] -= 1
                    break
                tail -= 1
            if tail < 0:
                continue
            row[tail] = b
            indeg[b] += 1
            tail -= 1
            replaced += 1
            written += 1
        table[ids[r]] = row
    return written


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

    @pytest.mark.parametrize("seed", range(40))
    def test_full_rows_match_sequential_reference(self, seed):
        """CAGRA graphs have full rows: the result is exactly the sequential one"""
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
            with_empty_slots=False,
        )

        expected = table.copy()
        expected_written = reference_projection(
            expected, layer1, promoted, max_share, protect
        )
        result, written = run_projection(table, promoted, layer1, max_share, protect)

        np.testing.assert_array_equal(result, expected)
        assert written == expected_written
        assert_invariants(table, result, promoted, max_share, protect)

    @pytest.mark.parametrize("seed", range(40, 70))
    def test_rows_with_empty_slots_keep_invariants(self, seed):
        """Rows with empty slots (not produced by CAGRA): every invariant holds, the same number of links is
        written, and without protection the result is exactly the sequential one"""
        rng = np.random.default_rng(seed)
        n = int(rng.integers(80, 1500))
        nb0 = int(rng.choice([8, 16, 32]))
        max_share = float(rng.choice([0.0, 0.25, 0.5, 1.0]))
        protect = bool(seed % 3 != 0)
        table, promoted, layer1 = random_case(
            rng,
            n,
            nb0,
            int(rng.choice([4, 8])),
            skewed=seed % 2 == 0,
            with_empty_slots=True,
        )

        expected = table.copy()
        expected_written = reference_projection(
            expected, layer1, promoted, max_share, protect
        )
        result, written = run_projection(table, promoted, layer1, max_share, protect)

        assert_invariants(table, result, promoted, max_share, protect)
        assert written == expected_written
        if not protect:
            np.testing.assert_array_equal(result, expected)

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
            project_upper_level(cpu_index, vectors, m=4, ef_construction=64)

        mock_draw.assert_called_once_with(n, 8)
        layer1_vectors, metric, links_per_node, ef_construction = mock_layer1.call_args[
            0
        ]
        assert layer1_vectors.dtype == np.float32
        assert layer1_vectors.flags.c_contiguous
        np.testing.assert_array_equal(
            layer1_vectors, vectors[promoted].astype(np.float32)
        )
        assert (metric, links_per_node, ef_construction) == (1, 8, 64)

        expected, written = run_projection(table, promoted, layer1, 0.5, True)
        assert written > 0
        np.testing.assert_array_equal(
            cpu_index.hnsw.neighbors.reshape(table.shape), expected
        )

    def test_fewer_than_two_promoted_is_a_no_op(self):
        table = np.array([[1], [0]], dtype=np.int32)
        cpu_index = self._cpu_index(table)

        with patch.object(
            projection, "draw_promoted", return_value=np.array([1], dtype=np.int32)
        ), patch.object(projection, "build_layer1") as mock_layer1:
            project_upper_level(
                cpu_index, np.zeros((2, 4), np.float32), m=16, ef_construction=100
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
