# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from abc import ABC, abstractmethod
from core.common.models import FaissIndexIDMap


class FaissCPUIndexBuilder(ABC):
    """
    Base class for CPU Index Configuration
    Also exposes methods to convert gpu index to cpu index from the configuration
    and writing cpu index to file
    """

    @abstractmethod
    def convert_gpu_to_cpu_index(
        self,
        gpu_index_id_map: FaissIndexIDMap,
    ) -> FaissIndexIDMap:
        """
        Implement this abstract method to convert a GPU vector search Index to a read compatible CPU Index

        Args:
        gpu_index_id_map (FaissIndexIDMap): A datamodel containing the GPU Faiss Index
        and dataset Vector Ids components

        Returns:
        FaissIndexIDMap: A datamodel containing the created CPU Faiss Index
        and dataset Vector Ids components
        """

        pass

    @abstractmethod
    def write_cpu_index(
        self, cpu_index_id_map: FaissIndexIDMap, cpu_index_output_file_path: str
    ) -> None:
        """
        Implement this abstract method to write the cpu index to specified output file path

        Args:
        cpu_index_id_map (FaissIndexIDMap): A datamodel containing the created GPU Faiss Index
        and dataset Vector Ids components
        cpu_index_output_file_path (str): File path to persist Index-Vector IDs map to
        """
        pass
