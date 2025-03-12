# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from abc import ABC, abstractmethod

from core.common.models import CPUIndexResponse


class CPUIndexWriter(ABC):
    """
    The CPUIndexWriter manages the process of writing a CPU vector search index to persistent storage
    """

    @abstractmethod
    def write_cpu_index(
        self, cpu_index_response: CPUIndexResponse, cpu_index_output_file_path: str
    ) -> None:
        """
        Implement this abstract method to write the cpu index to specified output file path
        """
        pass
