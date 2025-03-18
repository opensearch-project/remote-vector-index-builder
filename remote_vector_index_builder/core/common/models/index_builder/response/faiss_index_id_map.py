# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
from dataclasses import dataclass


@dataclass
class FaissIndexIDMap:
    """
    A data class that serves as a wrapper
    to hold the Faiss Index and dataset Vector Ids components
    which are written to persistent storage at the end of the index build process.
    """

    index_id_map: faiss.IndexIDMap

    def __del__(self):
        """
        Destructor to clean up FAISS resources.
        Ensures memory is properly freed when the object is destroyed.

        The method handles cleanup by
        explicitly deleting the internal Index and Vectors data
        """
        try:
            if self.index_id_map:
                if self.index_id_map.index:
                    # Delete the internal Index
                    self.index_id_map.index.thisown = True
                    del self.index_id_map.index
                # Delete the vectors, vector ids stored as part of the IndexIDMap
                self.index_id_map.own_fields = True
                del self.index_id_map
        except Exception as e:
            print(f"Error during cleanup of FaissGPUIndexResponse: {e}")
