# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
from typing import Dict, Any
from dataclasses import dataclass, field

from core.common.models import GraphBuildAlgo
from core.common.models import GPUIndexConfig
from .ivf_pq_build_cagra_config import IVFPQBuildCagraConfig
from .ivf_pq_search_cagra_config import IVFPQSearchCagraConfig


@dataclass
class GPUIndexCagraConfig(GPUIndexConfig):
    """
    Configuration class for Faiss GPU Index Cagra
    """

    # Degree of input graph for pruning
    intermediate_graph_degree: int = 128
    # Degree of output graph
    graph_degree: int = 64
    # ANN Algorithm to build the knn graph
    graph_build_algo: GraphBuildAlgo = GraphBuildAlgo.IVF_PQ

    store_dataset: bool = True

    # GPU Device on which the index is resident
    device: int = 0

    ivf_pq_build_config: IVFPQBuildCagraConfig = field(
        default_factory=IVFPQBuildCagraConfig
    )

    ivf_pq_search_config: IVFPQSearchCagraConfig = field(
        default_factory=IVFPQSearchCagraConfig
    )

    def _configure_build_algo(self):
        """
        Maps the graph building algorithm enum to the corresponding FAISS implementation.

        Args:
            graph_build_algo: The algorithm type to use for building the graph

        Returns:
            The corresponding FAISS graph building algorithm implementation
            Defaults to IVF_PQ if the specified algorithm is not found
        """
        switcher = {GraphBuildAlgo.IVF_PQ: faiss.graph_build_algo_IVF_PQ}
        return switcher.get(self.graph_build_algo, faiss.graph_build_algo_IVF_PQ)

    def _validate_params(params: Dict[str, Any]) -> None:
        """
        Pre-validates GPUIndexCagraConfig configuration parameters before object creation.

        Args:
            params: Dictionary of parameters to validate

        Raises:
            ValueError: If any parameter fails validation
        """
        if "intermediate_graph_degree" in params:
            if params["intermediate_graph_degree"] <= 0:
                raise ValueError(
                    "GPUIndexCagraConfig param: intermediate_graph_degree must be positive"
                )

        if "graph_degree" in params:
            if params["graph_degree"] <= 0:
                raise ValueError(
                    "GPUIndexCagraConfig param: graph_degree must be positive"
                )

        if "device" in params:
            if params["device"] < 0:
                raise ValueError(
                    "GPUIndexCagraConfig param: device must be non-negative"
                )

    def to_faiss_config(self) -> faiss.GpuIndexCagraConfig:
        """
        Builds and returns the complete faiss.GPUIndexCagraConfig
        Configures -
        - Basic GPUIndex Cagra Config parameters
        - IVF-PQ Build Cagra Config parameters
        - IVF-PQ Search Cagra Config paramters

        Returns:
            A fully configured faiss.GPUIndexCagraConfig object ready for index creation
        """

        gpu_index_cagra_config = faiss.GpuIndexCagraConfig()

        # Set basic parameters
        gpu_index_cagra_config.intermediate_graph_degree = (
            self.intermediate_graph_degree
        )
        gpu_index_cagra_config.graph_degree = self.graph_degree
        gpu_index_cagra_config.store_dataset = self.store_dataset
        gpu_index_cagra_config.device = self.device

        # Set build algorithm
        gpu_index_cagra_config.build_algo = self._configure_build_algo()

        if self.graph_build_algo == GraphBuildAlgo.IVF_PQ:
            gpu_index_cagra_config.ivf_pq_build_params = (
                self.ivf_pq_build_config.to_faiss_config()
            )
            gpu_index_cagra_config.ivf_pq_search_params = (
                self.ivf_pq_search_config.to_faiss_config()
            )

        return gpu_index_cagra_config

    @classmethod
    def from_dict(cls, params: Dict[str, Any] | None = None) -> "GPUIndexCagraConfig":
        """
        Constructs a GPUIndexCagraConfig object from a dictionary of parameters.

        Args:
            params: A dictionary containing the configuration parameters

        Returns:
            A GPUIndexCagraConfig object with the specified configuration
        """
        if not params:
            return cls()

        # Extract and configure IVF-PQ build parameters
        ivf_pq_build_params = params.pop("ivf_pq_build_params", None)
        ivf_pq_build_config = IVFPQBuildCagraConfig.from_dict(ivf_pq_build_params)

        # Extract and configure IVF-PQ search parameters
        ivf_pq_search_params = params.pop("ivf_pq_search_params", None)
        ivf_pq_search_config = IVFPQSearchCagraConfig.from_dict(ivf_pq_search_params)

        # Extract and configure graph build algo enum
        if "graph_build_algo" in params:
            params["graph_build_algo"] = GraphBuildAlgo(params["graph_build_algo"])

        # Validate parameters
        cls._validate_params(params)

        # Create and set the complete GPUIndexCagraConfig
        return cls(
            **params,
            ivf_pq_build_config=ivf_pq_build_config,
            ivf_pq_search_config=ivf_pq_search_config
        )
