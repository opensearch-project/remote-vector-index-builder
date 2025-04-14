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


def get_omp_num_threads():
    """
    Calculate the number of OpenMP threads to use for parallel processing.
    Returns the maximum of (CPU count - 2) or 1 to ensure at least one thread.

    Returns:
        int: Number of threads to use
    """
    return max(math.floor(os.cpu_count() - 2), 1)


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
