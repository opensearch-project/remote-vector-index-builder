from typing import Any, Dict, Optional
from remote_vector_index_builder.core.common.models import (
    IndexHNSWCagraConfig,
    GPUIndexCagraConfig,
    SpaceType,
    IVFPQBuildCagraConfig,
    IVFPQSearchCagraConfig,
    GraphBuildAlgo,
    GPUIndexBuildConfig,
)


class IndexConfigBuilder:
    def __init__(self):
        self._hnsw_config: Optional[IndexHNSWCagraConfig] = None
        self._gpu_config: Optional[GPUIndexCagraConfig] = None
        self._metric: SpaceType = SpaceType("l2")  # default metric

    def set_hnsw_config(self, params: Dict[str, Any]) -> "IndexConfigBuilder":
        self._hnsw_config = (
            IndexHNSWCagraConfig(**params) if params else IndexHNSWCagraConfig()
        )
        return self

    def set_gpu_config(self, params: Dict[str, Any]) -> "IndexConfigBuilder":
        if not params:
            self._gpu_config = GPUIndexCagraConfig()
            return self

        ivf_pq_build_params = params.pop("ivf_pq_build_params", None)
        ivf_pq_build_config = (
            IVFPQBuildCagraConfig(**ivf_pq_build_params)
            if ivf_pq_build_params
            else IVFPQBuildCagraConfig()
        )

        ivf_pq_search_params = params.pop("ivf_pq_search_params", None)
        ivf_pq_search_config = (
            IVFPQSearchCagraConfig(**ivf_pq_search_params)
            if ivf_pq_search_params
            else IVFPQSearchCagraConfig()
        )

        graph_build_algo_param = params.pop("graph_build_algo", None)
        graph_build_algo = (
            GraphBuildAlgo(graph_build_algo_param)
            if graph_build_algo_param
            else GraphBuildAlgo.IVF_PQ
        )

        self._gpu_config = GPUIndexCagraConfig(
            **params,
            graph_build_algo=graph_build_algo,
            ivf_pq_build_config=ivf_pq_build_config,
            ivf_pq_search_config=ivf_pq_search_config
        )
        return self

    def set_metric(self, metric: str) -> "IndexConfigBuilder":
        self._metric = SpaceType(metric)
        return self

    def build(self) -> GPUIndexBuildConfig:
        if not self._hnsw_config:
            self._hnsw_config = IndexHNSWCagraConfig()
        if not self._gpu_config:
            self._gpu_config = GPUIndexCagraConfig()

        return GPUIndexBuildConfig(
            index_hnsw_cagra_config=self._hnsw_config,
            gpu_index_cagra_config=self._gpu_config,
            metric=self._metric,
        )
