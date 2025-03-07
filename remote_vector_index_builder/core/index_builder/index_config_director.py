from typing import Dict, Any
from core.common.models import GPUIndexBuildConfig
from core.index_builder.index_config_builder import IndexConfigBuilder


class IndexConfigDirector:
    """Director class to construct index configurations using the builder"""

    def __init__(self, builder: IndexConfigBuilder):
        """
        Initialize the director with a config builder.

        Args:
            builder (IndexConfigBuilder): The builder instance to use for constructing configurations
        """
        self._builder = builder

    def construct_config(self, config_params: Dict[str, Any]) -> GPUIndexBuildConfig:
        """
        Constructs a GPU index build configuration using the provided parameters.

        Uses builder pattern to set various configuration options with default values
        if not specified in the input parameters.

        Args:
            config_paams (Dict[str, Any]): Dictionary containing configuration parameters
            - hnsw_config: Configures IndexHNSWCagraConfig core datamodel
            - gpu_config: Configures GPUIndexCagraConfig core datamodel
            - metric: Distance metric to used during GPUIndex creation (defaults to 'l2')

        Returns:
            GPUIndexBuildConfig: A fully constructed Index Build config object
        """
        return (
            self._builder.set_hnsw_config(config_params.get("hnsw_config", {}))
            .set_gpu_config(config_params.get("gpu_config", {}))
            .set_metric(config_params.get("metric"))
            .build()
        )
