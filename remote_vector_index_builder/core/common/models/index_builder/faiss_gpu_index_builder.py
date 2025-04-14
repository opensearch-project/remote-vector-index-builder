# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from abc import ABC, abstractmethod

from core.common.models import (
    SpaceType,
    VectorsDataset,
)

from core.common.models.index_builder import (
    FaissGpuBuildIndexOutput,
)


class FaissGPUIndexBuilder(ABC):
    """
    Base class for GPU Index configurations
    Also exposes a method to build the gpu index from the configuration
    """

    # GPU Device on which the index is resident
    device: int = 0

    @abstractmethod
    def build_gpu_index(
        self,
        vectorsDataset: VectorsDataset,
        dataset_dimension: int,
        space_type: SpaceType,
    ) -> FaissGpuBuildIndexOutput:
        """
        Implement this abstract method to build a GPU Index for the specified vectors dataset

        Args:
        vectorsDataset (VectorsDataset): VectorsDataset object containing vectors and document IDs
        dataset_dimension (int): Dimension of the vectors
        space_type (SpaceType, optional): Distance metric to be used (defaults to L2)

        Returns:
        FaissGpuBuildIndexOutput: A data model containing the created GPU Faiss Index and dataset Vector Ids components
        """

        pass
