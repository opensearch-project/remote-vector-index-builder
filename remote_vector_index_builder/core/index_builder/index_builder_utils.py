# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import os
import math
import faiss


from core.common.models import (
    SpaceType,
)

# CAGRA graph degree and intermediate graph degree are both m * GRAPH_DEGREE_PER_M
GRAPH_DEGREE_PER_M = 4


def get_omp_num_threads():
    """
    Calculate the number of OpenMP threads to use for parallel processing.
    Returns the maximum of (CPU count/4) or 1 to ensure at least one thread.

    Returns:
        int: Number of threads to use
    """
    return max(math.floor(os.cpu_count() / 4), 1)


def calculate_ivf_pq_n_lists(doc_count: int):
    """
    Calculate the number of lists/clusters for IVF (Inverted File) index.
    Uses square root of document count as a heuristic.
    Returns a the rounded down square root of the doc_count

    Args:
        doc_count (int): Total number of documents

    Returns:
        int: Number of lists/clusters to use
    """
    return int(math.sqrt(doc_count))


def calculate_effective_m(m: int, doc_count: int, n_lists: int, n_probes: int) -> int:
    """
    Lower m for datasets too small for the requested m, so the CAGRA graph degree
    (m * GRAPH_DEGREE_PER_M) never asks for more neighbors than the dataset can supply.

    CAGRA builds its initial kNN graph with an IVF-PQ search that probes `n_probes` of `n_lists`
    lists, so each vector sees roughly n_probes * doc_count / n_lists candidates. When the graph
    degree is close to or above that, the initial graph is left with missing or repeated
    neighbors and cuVS rejects it ("too many invalid or duplicated neighbor nodes"). The degree
    is kept within half of the candidates, and below doc_count. Datasets that can already supply
    the requested m are unaffected.

    Args:
        m (int): Requested HNSW m
        doc_count (int): Total number of documents
        n_lists (int): Number of IVF-PQ lists used for the build
        n_probes (int): Number of IVF-PQ lists probed per vector

    Returns:
        int: m to use for this build, between 1 and the requested m
    """
    n_lists = max(n_lists, 1)
    candidates = min(n_probes, n_lists) * doc_count // n_lists
    max_graph_degree = min(candidates // 2, doc_count - 1)
    return max(1, min(m, max_graph_degree // GRAPH_DEGREE_PER_M))


def configure_metric(space_type: SpaceType):
    """
    Map SpaceType to corresponding FAISS distance metric.

    Args:
        space_type (SpaceType): Type of vector space metric to use

    Returns:
        int: FAISS metric constant
    """
    switcher = {
        SpaceType.L2: faiss.METRIC_L2,
        SpaceType.INNERPRODUCT: faiss.METRIC_INNER_PRODUCT,
    }
    return switcher.get(space_type, faiss.METRIC_L2)
