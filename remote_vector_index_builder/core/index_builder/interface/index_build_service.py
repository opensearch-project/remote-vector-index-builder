# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from abc import ABC, abstractmethod
from typing import Any
from core.common.models import VectorsDataset, IndexBuildParameters


class IndexBuildService(ABC):
    """
    The Index Build Service orchestrates the workflow of building a vector search index
    New engines extending this class must call the necessary workflow steps in the build_index method
    """

    @abstractmethod
    def build_index(
        self,
        index_build_parameters: IndexBuildParameters,
        vectors_dataset: VectorsDataset,
    ) -> Any:
        """
        Implement this abstract method orchestrating an index build for the specified vectors dataset
        and input index build parameters
        """
        pass
