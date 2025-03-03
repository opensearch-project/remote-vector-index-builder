# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from dataclasses import dataclass


@dataclass
class IVFPQBuildCagraConfig:
    # The number of inverted lists (clusters)
    # Hint: the number of vectors per cluster (`n_rows/n_lists`) should be
    # approximately 1,000 to 10,000.
    n_lists: int = 1000

    # The number of iterations searching for kmeans centers (index building).
    kmeans_n_iters: int = 10
    # The fraction of data to use during iterative kmeans building.
    kmeans_trainset_fraction: float = 0.1

    # The bit length of the vector element after compression by PQ.
    # Possible values: [4, 5, 6, 7, 8].
    # Hint: the smaller the 'pq_bits', the smaller the index size and the
    # better the search performance, but the lower the recall.
    pq_bits: int = 8

    # The dimensionality of the vector after compression by PQ. When zero, an
    # optimal value is selected using a heuristic.
    # pq_bits` must be a multiple of 8.
    # Hint: a smaller 'pq_dim' results in a smaller index size and better
    # search performance, but lower recall. If 'pq_bits' is 8, 'pq_dim' can be
    # set to any number, but multiple of 8 are desirable for good performance.
    # If 'pq_bits' is not 8, 'pq_dim' should be a multiple of 8. For good
    # performance, it is desirable that 'pq_dim' is a multiple of 32
    # Ideally 'pq_dim' should be also a divisor of the dataset dim.
    pq_dim: int = 16

    # By default, the algorithm allocates more space than necessary for
    # individual clusters
    # This allows to amortize the cost of memory allocation and
    # reduce the number of data copies during repeated calls to `extend`
    # (extending the database).
    #
    # The alternative is the conservative allocation behavior; when enabled,
    # the algorithm always allocates the minimum amount of memory required to
    # store the given number of records. Set this flag to `true` if you prefer
    # to use as little GPU memory for the database as possible.
    conservative_memory_allocation: bool = True
