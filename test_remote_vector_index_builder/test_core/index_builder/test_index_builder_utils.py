# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import pytest

from core.index_builder.index_builder_utils import (
    GRAPH_DEGREE_PER_M,
    calculate_effective_m,
    calculate_ivf_pq_n_lists,
)

N_PROBES = 5


def _effective_m(m, doc_count):
    return calculate_effective_m(
        m, doc_count, calculate_ivf_pq_n_lists(doc_count), N_PROBES
    )


@pytest.mark.parametrize(
    "m,doc_count,expected",
    [
        (16, 5, 1),  # smallest build k-NN sends
        (16, 131, 7),  # segment sizes that failed with m=16
        (16, 192, 9),
        (16, 400, 12),
        (16, 1_000, 16),  # can already supply m * 4 neighbors: requested m is kept
        (16, 1_000_000, 16),
        (4, 131, 4),  # small m already fits
        (128, 1_000, 20),  # large m on a dataset that cannot supply it is lowered too
        (128, 1_000_000, 128),
    ],
)
def test_calculate_effective_m(m, doc_count, expected):
    assert _effective_m(m, doc_count) == expected


@pytest.mark.parametrize("doc_count", [5, 6, 7, 8, 10, 25, 50, 131, 192, 400, 1_000])
def test_effective_graph_degree_fits_the_dataset(doc_count):
    n_lists = calculate_ivf_pq_n_lists(doc_count)
    degree = _effective_m(16, doc_count) * GRAPH_DEGREE_PER_M
    candidates = min(N_PROBES, n_lists) * doc_count // n_lists

    # Never more neighbors than there are other vectors
    assert degree <= doc_count - 1
    # Never more than the IVF-PQ search can see, except at the minimum m of 1
    assert degree <= candidates or degree == GRAPH_DEGREE_PER_M


def test_effective_m_never_exceeds_requested_or_drops_below_one():
    for doc_count in (5, 50, 500, 5_000, 500_000):
        for m in (1, 2, 16, 64):
            assert 1 <= _effective_m(m, doc_count) <= m
