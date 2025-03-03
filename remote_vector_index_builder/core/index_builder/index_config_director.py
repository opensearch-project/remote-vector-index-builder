from typing import Dict, Any
from remote_vector_index_builder.core.common.models.index_builder.gpu_index_build_config import (
    GPUIndexBuildConfig,
)
from remote_vector_index_builder.core.index_builder.index_config_builder import (
    IndexConfigBuilder,
)


class IndexConfigDirector:
    """Director class to construct index configurations using the builder"""

    def __init__(self, builder: IndexConfigBuilder):
        self._builder = builder

    def construct_config(self, config_params: Dict[str, Any]) -> GPUIndexBuildConfig:
        return (
            self._builder.set_hnsw_config(config_params.get("hnsw_config", {}))
            .set_gpu_config(config_params.get("gpu_config", {}))
            .set_metric(config_params.get("metric", "l2"))
            .build()
        )
