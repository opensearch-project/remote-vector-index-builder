# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
from dataclasses import dataclass


@dataclass
class FaissGpuBuildIndexOutput:
    """
    A data class that serves as a wrapper
    to hold the Faiss GPU Index and dataset Vector Ids components
    """

    gpu_index: faiss.GpuIndexCagra
    index_id_map: faiss.IndexIDMap

    def __del__(self):
        """
        Destructor to clean up FAISS resources.
        Ensures memory is properly freed when the object is destroyed.

        The method handles cleanup by
        explicitly deleting the internal Index and Vectors data
        """
        try:
            if self.gpu_index:
                # Delete the internal Index
                del self.gpu_index
            if self.index_id_map:
                # Delete the vectors, vector ids stored as part of the IndexIDMap
                self.index_id_map.own_fields = True
                del self.index_id_map
        except Exception as e:
            print(f"Error during cleanup of FaissGpuBuildIndexOutput: {e}")
