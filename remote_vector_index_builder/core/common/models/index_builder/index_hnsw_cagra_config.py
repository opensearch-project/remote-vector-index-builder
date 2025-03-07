# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from dataclasses import dataclass


@dataclass
class IndexHNSWCagraConfig:
    # expansion factor at search time
    ef_search: int = 256

    # expansion factor at construction time
    ef_construction: int = 40

    # When set to true, the index is immutable.
    # This option is used to copy the knn graph from GpuIndexCagra
    # to the base level of IndexHNSWCagra without adding upper levels.
    # Doing so enables to search the HNSW index, but removes the
    # ability to add vectors.
    base_level_only: bool = True

    # Set to true to delete internal storage:Index variable
    # when destructor is called
    own_fields: bool = True
