# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from dataclasses import dataclass
from typing import Dict, Any
from core.common.models import CPUIndexConfig


@dataclass
class IndexHNSWCagraConfig(CPUIndexConfig):
    """Configuration class for HNSW Cagra CPU Index"""

    # expansion factor at search time
    ef_search: int = 100

    # expansion factor at construction time
    ef_construction: int = 100

    # When set to true, the index is immutable.
    # This option is used to copy the knn graph from GpuIndexCagra
    # to the base level of IndexHNSWCagra without adding upper levels.
    # Doing so enables to search the HNSW index, but removes the
    # ability to add vectors.
    base_level_only: bool = True

    @classmethod
    def from_dict(cls, params: Dict[str, Any] | None = None) -> "IndexHNSWCagraConfig":
        """
        Constructs an IndexHNSWCagraConfig object from a dictionary of parameters.

        Args:
            params: A dictionary containing the configuration parameters

        Returns:
            An instance of IndexHNSWCagraConfig with the specified parameters
        """
        if not params:
            return cls()

        return cls(**params)
