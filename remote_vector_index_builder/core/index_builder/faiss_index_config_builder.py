from typing import Optional
import faiss
from core.common.models import (
    GPUIndexCagraConfig,
    IVFPQBuildCagraConfig,
    IVFPQSearchCagraConfig,
    GraphBuildAlgo,
)


class FaissIndexConfigBuilder:
    """
    A builder class for configuring FAISS GPU indexes

    This class helps construct configuration objects for GPU-based FAISS indexes, including:
    - GPU index Cagra configuration
    - IVF-PQ (Inverted File System with Product Quantization) Cagra build parameters
    - IVF-PQ Cagra search parameters
    """

    def __init__(self):
        """
        Initialize the builder with default configuration values set to None.
        These configurations will be set later using builder methods.
        """
        self._gpu_config: Optional[GPUIndexCagraConfig] = None
        self._ivf_pq_build_config: Optional[IVFPQBuildCagraConfig] = None
        self._ivf_pq_search_config: Optional[IVFPQSearchCagraConfig] = None

    def _configure_build_algo(self, graph_build_algo: GraphBuildAlgo):
        """
        Maps the graph building algorithm enum to the corresponding FAISS implementation.

        Args:
            graph_build_algo: The algorithm type to use for building the graph

        Returns:
            The corresponding FAISS graph building algorithm implementation
            Defaults to IVF_PQ if the specified algorithm is not found
        """
        switcher = {GraphBuildAlgo.IVF_PQ: faiss.graph_build_algo_IVF_PQ}
        return switcher.get(graph_build_algo, faiss.graph_build_algo_IVF_PQ)

    def _create_ivf_pq_build_config(self) -> faiss.IVFPQBuildCagraConfig:
        """
        Creates and configures the equivalent FAISS IVFPQBuildCagraConfig from the
        IVFPQBuildCagraConfig core datamodel.

        Returns:
             A configured FAISS IVFPQBuildCagraConfig object with parameters for:
            - kmeans training set fraction
            - kmeans iteration count
            - Product Quantization bits and dimensions
            - Number of inverted lists (kmeans clusters)
            - Memory allocation strategy
        """

        config = faiss.IVFPQBuildCagraConfig()
        config.kmeans_trainset_fraction = (
            self._ivf_pq_build_config.kmeans_trainset_fraction
        )
        config.kmeans_n_iters = self._ivf_pq_build_config.kmeans_n_iters
        config.pq_bits = self._ivf_pq_build_config.pq_bits
        config.pq_dim = self._ivf_pq_build_config.pq_dim
        config.n_lists = self._ivf_pq_build_config.n_lists
        config.conservative_memory_allocation = (
            self._ivf_pq_build_config.conservative_memory_allocation
        )
        return config

    def _create_ivf_pq_search_config(self) -> faiss.IVFPQSearchCagraConfig:
        """
        Creates and configures the equivalent FAISS IVFPQSearchCagraConfig from the
        IVFPQSearchCagraConfig core datamodel.
        Returns:
            A configured FAISS IVFPQSearchCagraConfig object with search parameters for:
            - n_probs The number of clusters to search
        """

        config = faiss.IVFPQSearchCagraConfig()
        config.n_probes = self._ivf_pq_search_config.n_probes
        return config

    def with_gpu_config(
        self, gpu_config: GPUIndexCagraConfig
    ) -> "FaissIndexConfigBuilder":
        """
        Sets the GPUIndexCagraConfig for the index builder.

        Args:
            gpu_config: GPUIndexCagraConfig core datamodel

        Returns:
            Self reference for method chaining
        """
        self._gpu_config = gpu_config
        if gpu_config:
            self._ivf_pq_build_config = gpu_config.ivf_pq_build_config
            self._ivf_pq_search_config = gpu_config.ivf_pq_search_config
        return self

    def build_gpu_index_cagra_config(self) -> faiss.GpuIndexCagraConfig:
        """
        Builds and returns the complete FAISS GPUIndexCagraConfig
        Configures -
        - Basic GPUIndex Cagra Config parameters
        - IVF-PQ Build Cagra Config parameters
        - IVF-PQ Search Cagra Config paramters

        Returns:
            A fully configured faiss GPUIndexCagraConfig object ready for index creation
        """
        if not self._gpu_config:
            self._gpu_config = GPUIndexCagraConfig()
            self._ivf_pq_build_config = self._gpu_config.ivf_pq_build_config
            self._ivf_pq_search_config = self._gpu_config.ivf_pq_search_config

        gpu_index_cagra_config = faiss.GpuIndexCagraConfig()
        gpu_index_cagra_config.intermediate_graph_degree = (
            self._gpu_config.intermediate_graph_degree
        )
        gpu_index_cagra_config.graph_degree = self._gpu_config.graph_degree
        gpu_index_cagra_config.store_dataset = self._gpu_config.store_dataset

        gpu_index_cagra_config.build_algo = self._configure_build_algo(
            self._gpu_config.graph_build_algo
        )
        gpu_index_cagra_config.device = self._gpu_config.device

        if self._ivf_pq_build_config:
            gpu_index_cagra_config.ivf_pq_build_params = (
                self._create_ivf_pq_build_config()
            )

        if self._ivf_pq_search_config:
            gpu_index_cagra_config.ivf_pq_search_params = (
                self._create_ivf_pq_search_config()
            )

        return gpu_index_cagra_config
