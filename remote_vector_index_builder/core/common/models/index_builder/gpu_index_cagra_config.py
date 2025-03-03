# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from dataclasses import dataclass, field

from .graph_build_algo import GraphBuildAlgo
from .ivf_pq_build_cagra_config import IVFPQBuildCagraConfig
from .ivf_pq_search_cagra_config import IVFPQSearchCagraConfig


@dataclass
class GPUIndexCagraConfig:
    # Degree of input graph for pruning
    intermediate_graph_degree: int = 64
    # Degree of output graph
    graph_degree: int = 32
    # ANN Algorithm to build the knn graph
    graph_build_algo: GraphBuildAlgo = GraphBuildAlgo.IVF_PQ

    store_dataset: bool = False
    # GPU Device on which the index is resident
    device: int = 0

    ivf_pq_build_config: IVFPQBuildCagraConfig = field(
        default_factory=IVFPQBuildCagraConfig
    )

    ivf_pq_search_config: IVFPQSearchCagraConfig = field(
        default_factory=IVFPQSearchCagraConfig
    )
