# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from abc import ABC, abstractmethod

from core.common.models import (
    VectorsDataset,
    SpaceType,
    GPUIndexConfig,
    GPUIndexResponse,
)


class GPUIndexBuildService(ABC):
    """
    The GPU Index Build Service manages the process of building a vector search index on a GPU
    """

    @abstractmethod
    def build_gpu_index(
        self,
        vectors_dataset: VectorsDataset,
        dataset_dimension: int,
        gpu_index_config: GPUIndexConfig,
        space_type: SpaceType,
    ) -> GPUIndexResponse:
        """
        Implement this abstract method to build a GPU Index for the specified vectors dataset
        """
        pass
