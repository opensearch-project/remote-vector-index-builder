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
        self.cleanup()

    def cleanup(self):
        """
        Clean up method for FAISS resources.
        Ensures memory is properly freed when the object is destroyed.

        The method handles cleanup by
        explicitly deleting the internal Index and Vectors data
        """
        try:
            # Cleanup the GPU Index explicitly if the CPU Index was successfully created
            # OR if the orchestrator fails after creating a GPU Index and
            # before the CPU Index is created, and replaced GPU Index in the IndexIDMap
            if self.gpu_index:
                # Delete the internal Index

                if self.index_id_map:
                    self.index_id_map.index = None
                self.gpu_index.thisown = True

                gpu_index = self.gpu_index
                self.gpu_index = None
                gpu_index.__swig_destroy__(gpu_index)

            # A reference to IndexIDMap is maintained until the underlying GPU Index is replaced by a CPU Index
            # This block runs if the orchestrator fails before the successful replacement
            if self.index_id_map:
                # Delete the vectors, vector ids stored as part of the IndexIDMap
                self.index_id_map.thisown = True
                self.index_id_map.own_fields = False

                self.index_id_map.index = None

                index_id_map = self.index_id_map
                self.index_id_map = None
                index_id_map.__swig_destroy__(index_id_map)
        except Exception as e:
            print(f"Error during cleanup of FaissGpuBuildIndexOutput: {e}")
