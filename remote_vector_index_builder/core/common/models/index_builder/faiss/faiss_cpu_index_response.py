# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
from dataclasses import dataclass

from core.common.models import CPUIndexResponse


@dataclass
class FaissCPUIndexResponse(CPUIndexResponse):
    """
    A data class that holds FAISS CPU vector search index components for vector similarity search.

    Attributes:
        cpu_index_cagra (faiss.IndexHNSWCagra): The CPU converted, GPU-accelerated CAGRA index for
            performing vector similarity searches.
        index_id_map: (faiss.Index): An index that maintains mapping between vectors
            and the cpu index
    """

    cpu_index: faiss.IndexHNSWCagra
    index_id_map: faiss.Index

    def __del__(self):
        """
        Destructor to clean up FAISS resources.
        Ensures CPU memory is properly freed when the object is destroyed.

        The method handles cleanup by
        explicitly deleting cpu index to free CPU and system memory
        """

        try:
            if self.cpu_index:
                self.cpu_index.own_fields = True
                del self.cpu_index
        except Exception as e:
            print(f"Error during cleanup of FaissCPUIndexResponse: {e}")
