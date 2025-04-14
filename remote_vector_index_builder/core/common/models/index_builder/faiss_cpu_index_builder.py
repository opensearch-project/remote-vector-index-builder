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
    and writing cpu index to file
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

    @abstractmethod
    def write_cpu_index(
        self,
        cpu_build_index_output: FaissCpuBuildIndexOutput,
        cpu_index_output_file_path: str,
    ) -> None:
        """
        Implement this abstract method to write the cpu index to specified output file path

        Args:
        cpu_build_index_output (FaissCpuBuildIndexOutput): A datamodel containing the created GPU Faiss Index
        and dataset Vector Ids components
        cpu_index_output_file_path (str): File path to persist Index-Vector IDs map to
        """
        pass
