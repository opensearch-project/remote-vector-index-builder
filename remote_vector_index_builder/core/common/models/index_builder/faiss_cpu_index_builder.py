# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from abc import ABC, abstractmethod
from core.common.models.index_builder import (
    FaissCpuBuildIndexOutput,
    FaissGpuBuildIndexOutput,
)


class FaissCPUIndexBuilder(ABC):
    """
    Base class for CPU Index Configuration
    Also exposes methods to convert gpu index to cpu index from the configuration
    """

    @abstractmethod
    def convert_gpu_to_cpu_index(
        self,
        gpu_build_index_output: FaissGpuBuildIndexOutput,
    ) -> FaissCpuBuildIndexOutput:
        """
        Implement this abstract method to convert a GPU vector search Index to a read compatible CPU Index

        Args:
        gpu_build_index_output (FaissGpuBuildIndexOutput): A datamodel containing the GPU Faiss Index
        and dataset Vector Ids components

        Returns:
        FaissCpuBuildIndexOutput: A datamodel containing the created CPU Faiss Index
        and dataset Vector Ids components
        """

        pass
