# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from abc import ABC, abstractmethod

from core.common.models import (
    CPUIndexConfig,
    CPUIndexResponse,
    GPUIndexResponse,
)


class GPUToCPUIndexConverter(ABC):
    """
    The GPUToCPUIndexConverter manages the process of converting a GPU vector search index to a CPU Index.
    """

    @abstractmethod
    def convert_gpu_to_cpu_index(
        self,
        gpu_index_response: GPUIndexResponse,
        cpu_index_config: CPUIndexConfig,
    ) -> CPUIndexResponse:
        """
        Implement this abstract method to convert a GPU vector search Index to a read compatible CPU Index
        """
        pass
