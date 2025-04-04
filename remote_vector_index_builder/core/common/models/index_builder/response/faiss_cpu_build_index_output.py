# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
from dataclasses import dataclass


@dataclass
class FaissCpuBuildIndexOutput:
    """
    A data class that serves as a wrapper
    to hold the Faiss CPU Index and dataset Vector Ids components
    which are written to persistent storage at the end of the index build process.
    """

    cpu_index: faiss.IndexHNSWCagra
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
            # Cleanup the CPU Index explicitly after it is succussfully persisted to disc
            # OR if the orchestrator fails after creating the CPU Index
            # and before the CPU Index is persisted to disc
            if self.cpu_index:
                # Delete the internal Index
                self.cpu_index.thisown = True

                cpu_index = self.cpu_index
                self.cpu_index = None
                cpu_index.__swig_destroy__(cpu_index)

            # Delete the IndexIDMap not containing any underlying Index
            # This block runs after the CPU Index is cleaned up
            # if the CPU Index is succussfully persisted to disc
            # OR if the orchestrator fails after creating the CPU Index
            # and before the CPU Index is persisted to disc
            if self.index_id_map:
                # Delete the vectors, vector ids stored as part of the IndexIDMap
                self.index_id_map.own_fields = False
                self.index_id_map.thisown = True

                index_id_map = self.index_id_map
                self.index_id_map = None
                index_id_map.__swig_destroy__(index_id_map)
        except Exception as e:
            print(f"Error during cleanup of FaissCpuBuildIndexOutput: {e}")
