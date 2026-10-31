# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
import logging
from dataclasses import dataclass
from typing import Dict, Any
from core.common.models.index_builder import (
    FaissCpuBuildIndexOutput,
    FaissGpuBuildIndexOutput,
    FaissCPUIndexBuilder,
)

from remote_vector_index_builder.core.common.models.index_build_parameters import (
    DataType,
)

logger = logging.getLogger(__name__)


@dataclass
class FaissIndexHNSWCagraBuilder(FaissCPUIndexBuilder):
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

    vector_dtype: DataType = DataType.FLOAT

    skip_stored_vectors: bool = False

    # When set to true (and base_level_only is true), copyTo also builds an
    # HNSW layer 1 over a promoted subset of the vectors and projects its links
    # into the base level, so graphs of well-separated clusters stay connected
    # (faiss patch 0004: GpuIndexCagra::copyTo(..., CagraUpperLevelProjectionParams)).
    # The index stays base-level only. Not supported for binary vectors.
    project_link_to_base_layer: bool = False

    # HNSW m of the index; required when project_link_to_base_layer is true. Layer 1 is
    # built like an HNSW with 2 * m links per node and promotion probability 1 / (2 * m).
    m: int = 0

    # efConstruction of the layer-1 build; 0 = ef_construction
    upper_level_ef_construction: int = 0

    # Max share of a full base-level row that may be overwritten by projected links
    upper_level_max_share: float = 0.5

    # Never overwrite a node's last base-level in-edge
    upper_level_protect_last_in_edge: bool = True

    @classmethod
    def from_dict(
        cls, params: Dict[str, Any] | None = None
    ) -> "FaissIndexHNSWCagraBuilder":
        """
        Constructs an FaissIndexHNSWCagraBuilder object from a dictionary of parameters.

        Args:
            params: A dictionary containing the configuration parameters

        Returns:
            An instance of FaissIndexHNSWCagraBuilder with the specified parameters
        """
        if not params:
            return cls()

        return cls(**params)

    def _upper_level_projection_params(self):
        """Builds the faiss.CagraUpperLevelProjectionParams for copyTo from this config."""
        if self.m <= 0:
            raise ValueError(
                "FaissIndexHNSWCagraBuilder: m must be positive when project_link_to_base_layer is true"
            )
        params = faiss.CagraUpperLevelProjectionParams()
        params.M = self.m
        params.ef_construction = self.upper_level_ef_construction
        params.max_share = self.upper_level_max_share
        params.protect_last_in_edge = self.upper_level_protect_last_in_edge
        return params

    def _do_convert_gpu_to_cpu_index(
        self, faiss_gpu_build_index_output: FaissGpuBuildIndexOutput
    ):
        try:
            # Initialize CPU Index
            cpu_index = faiss.IndexHNSWCagra()

            # Configure CPU Index parameters
            cpu_index.hnsw.efConstruction = self.ef_construction
            cpu_index.hnsw.efSearch = self.ef_search
            cpu_index.base_level_only = self.base_level_only

            # Copy GPU index to CPU index
            if self.project_link_to_base_layer and not self.base_level_only:
                logger.warning(
                    "project_link_to_base_layer is only supported with base_level_only=True; "
                    "copying without projection"
                )
            if self.project_link_to_base_layer and self.base_level_only:
                stats = faiss.CagraUpperLevelProjectionStats()
                faiss_gpu_build_index_output.gpu_index.copyTo(
                    cpu_index,
                    self.skip_stored_vectors,
                    self._upper_level_projection_params(),
                    stats,
                )
                logger.debug(
                    f"Upper-level projection: promoted={stats.promoted} "
                    f"added_by_overwrite={stats.added_by_overwrite} "
                    f"added_into_empty={stats.added_into_empty} "
                    f"skipped_cap={stats.skipped_cap} "
                    f"protected_slots_skipped={stats.protected_slots_skipped} "
                    f"in_degree0={stats.in_degree0_before}->{stats.in_degree0_after} "
                    f"layer1_build_ms={stats.layer1_build_ms:.1f} "
                    f"projection_ms={stats.projection_ms:.1f}"
                )
            elif self.skip_stored_vectors:
                logger.debug(
                    "skip_stored_vectors=True: skipping vector storage during copyTo"
                )
                faiss_gpu_build_index_output.gpu_index.copyTo(cpu_index, True)
            else:
                faiss_gpu_build_index_output.gpu_index.copyTo(cpu_index)

            # Remove reference of GPU Index from the IndexIDMap
            faiss_gpu_build_index_output.index_id_map.index = None

            # Update the ID map index with the CPU index
            index_id_map = faiss_gpu_build_index_output.index_id_map

            # Remove reference of the IndexIDMap from the GPU Build Index Output before cleanup
            faiss_gpu_build_index_output.index_id_map = None

            index_id_map.index = cpu_index

            # Free memory taken by GPU Index
            faiss_gpu_build_index_output.cleanup()

            return FaissCpuBuildIndexOutput(
                cpu_index=cpu_index, index_id_map=index_id_map
            )
        except Exception as e:
            raise Exception(
                f"Failed to convert GPU index to CPU index: {str(e)}"
            ) from e

    def _do_convert_gpu_to_cpu_binary_index(
        self, faiss_gpu_build_index_output: FaissGpuBuildIndexOutput
    ):
        try:
            # Convert GPU binary index to CPU binary index
            cpu_index = faiss.IndexBinaryHNSWCagra()
            cpu_index.hnsw.efConstruction = self.ef_construction
            cpu_index.hnsw.efSearch = self.ef_search
            cpu_index.base_level_only = self.base_level_only

            # Convert GPU binary index to CPU binary index
            if self.skip_stored_vectors:
                logger.debug(
                    "skip_stored_vectors=True: skipping vector storage during copyTo"
                )
                faiss_gpu_build_index_output.gpu_index.copyTo(cpu_index, True)
            else:
                faiss_gpu_build_index_output.gpu_index.copyTo(cpu_index)

            # Remove reference of GPU Index from the IndexBinaryIDMap
            faiss_gpu_build_index_output.index_id_map.index = None

            # Update the ID map index with the CPU index
            index_id_map = faiss_gpu_build_index_output.index_id_map

            # Remove reference of the IndexBinaryIDMap from the GPU Build Index Output before cleanup
            faiss_gpu_build_index_output.index_id_map = None

            index_id_map.index = cpu_index

            # Free memory taken by GPU Index
            faiss_gpu_build_index_output.cleanup()

            return FaissCpuBuildIndexOutput(
                cpu_index=cpu_index, index_id_map=index_id_map
            )
        except Exception as e:
            raise Exception(
                f"Failed to convert GPU index to CPU index: {str(e)}"
            ) from e

    def convert_gpu_to_cpu_index(
        self, faiss_gpu_build_index_output: FaissGpuBuildIndexOutput
    ) -> FaissCpuBuildIndexOutput:
        """
        Method to convert a GPU Vector Search Index to CPU Index
        Returns a CPU read compatible vector search index
        Uses faiss specific library methods to achieve this.

        Args:
        faiss_gpu_build_index_output (FaissGpuBuildIndexOutput) A datamodel containing the GPU Faiss Index
        and dataset Vector Ids components

        Returns:
        FaissCpuBuildIndexOutput: A datamodel containing the created CPU Faiss Index
        and dataset Vector Ids components
        """

        if self.vector_dtype != DataType.BINARY:
            return self._do_convert_gpu_to_cpu_index(faiss_gpu_build_index_output)

        return self._do_convert_gpu_to_cpu_binary_index(faiss_gpu_build_index_output)
