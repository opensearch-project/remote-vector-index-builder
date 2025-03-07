from typing import Any, Dict, Optional
from core.common.models import (
    IndexHNSWCagraConfig,
    GPUIndexCagraConfig,
    SpaceType,
    IVFPQBuildCagraConfig,
    IVFPQSearchCagraConfig,
    GraphBuildAlgo,
    GPUIndexBuildConfig,
)


class IndexConfigBuilder:
    """
    A builder class for creating GPU index configurations with customizable parameters.
    Supports HNSW and GPU-specific configurations
    """

    def __init__(self):
        """
        Initialize the IndexConfigBuilder with default configurations.
        HNSW and GPU configs are initially set to None, with L2 as the default metric.
        """
        self._hnsw_config: Optional[IndexHNSWCagraConfig] = None
        self._gpu_config: Optional[GPUIndexCagraConfig] = None
        self._metric: SpaceType = SpaceType("l2")  # default metric

    def set_hnsw_config(self, params: Dict[str, Any]) -> "IndexConfigBuilder":
        """
        Set the HNSW config parameters for the CPU Index created

        Args:
            params: Dictionary containing HNSW configuration parameters.
                   If None, uses default configuration.

        Returns:
            self: Returns the builder instance for method chaining.
        """
        self._hnsw_config = (
            IndexHNSWCagraConfig(**params) if params else IndexHNSWCagraConfig()
        )
        return self

    def set_gpu_config(self, params: Dict[str, Any]) -> "IndexConfigBuilder":
        """
        Set the GPUIndexCagraConfig config parameters including IVF-PQ settings.

        Args:
            params: Dictionary containing GPUIndexCagraConfig parameters including:
                   - ivf_pq_build_params: Parameters for IVF-PQ index building
                   - ivf_pq_search_params: Parameters for IVF-PQ search
                   - graph_build_algo: Algorithm used for graph building

        Returns:
            self: Returns the builder instance for method chaining.
        """
        if not params:
            self._gpu_config = GPUIndexCagraConfig()
            return self

        # Extract and configure IVF-PQ build parameters
        ivf_pq_build_params = params.pop("ivf_pq_build_params", None)
        ivf_pq_build_config = (
            IVFPQBuildCagraConfig(**ivf_pq_build_params)
            if ivf_pq_build_params
            else IVFPQBuildCagraConfig()
        )

        # Extract and configure IVF-PQ search parameters
        ivf_pq_search_params = params.pop("ivf_pq_search_params", None)
        ivf_pq_search_config = (
            IVFPQSearchCagraConfig(**ivf_pq_search_params)
            if ivf_pq_search_params
            else IVFPQSearchCagraConfig()
        )

        # Extract and configure graph build algo enum
        graph_build_algo_param = params.pop("graph_build_algo", None)
        graph_build_algo = (
            GraphBuildAlgo(graph_build_algo_param)
            if graph_build_algo_param
            else GraphBuildAlgo.IVF_PQ
        )

        # Create and set the complete GPU Index Config
        self._gpu_config = GPUIndexCagraConfig(
            **params,
            graph_build_algo=graph_build_algo,
            ivf_pq_build_config=ivf_pq_build_config,
            ivf_pq_search_config=ivf_pq_search_config
        )
        return self

    def set_metric(self, metric: SpaceType) -> "IndexConfigBuilder":
        """
        Set the distance metric enum type SpaceType for similarity calculations.

        Args:
            metric (SpaceType): SpaceType enum representing the distance metric

        Returns:
            self: Returns the builder instance for method chaining.
        """
        self._metric = metric
        return self

    def build(self) -> GPUIndexBuildConfig:
        """
        Build and return the final GPU index configuration.
        If HNSW or GPU configs are not set, uses default configurations.

        Returns:
            GPUIndexBuildConfig: The complete index configuration object.
        """
        if not self._hnsw_config:
            self._hnsw_config = IndexHNSWCagraConfig()
        if not self._gpu_config:
            self._gpu_config = GPUIndexCagraConfig()

        return GPUIndexBuildConfig(
            index_hnsw_cagra_config=self._hnsw_config,
            gpu_index_cagra_config=self._gpu_config,
            metric=self._metric,
        )
