# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
from core.common.models.index_builder import (
    FaissCpuBuildIndexOutput,
    FaissGpuBuildIndexOutput,
    FaissCPUIndexBuilder,
)

from remote_vector_index_builder.core.common.models.index_build_parameters import (
    DataType,
)
from core.index_builder.faiss.cagra_upper_level_projection import (
    project_upper_level,
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

    # When set to true (and base_level_only is true), the conversion also builds
    # an HNSW layer 1 over a promoted subset of the vectors and projects its links
    # into the base level, so graphs of well-separated clusters stay connected
    # (see core.index_builder.faiss.cagra_upper_level_projection).
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

    # Skip the projection when at least this share of the nodes is already reachable from a
    # random entry point (well-connected graph); > 1 always projects
    upper_level_skip_if_reachable: float = 0.99

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

    def _project_link_to_base_layer(self, cpu_index, vectors: Optional[np.ndarray]):
        """Projects HNSW layer-1 links into the base level of cpu_index (in place)."""
        if self.m <= 0:
            raise ValueError(
                "FaissIndexHNSWCagraBuilder: m must be positive when project_link_to_base_layer is true"
            )
        if vectors is None:
            raise ValueError(
                "FaissIndexHNSWCagraBuilder: the dataset vectors are required when "
                "project_link_to_base_layer is true"
            )
        # Fail-safe: the projection only improves connectivity, so on any error the plain CAGRA graph
        # (left unchanged by project_upper_level) is kept and the build continues.
        try:
            project_upper_level(
                cpu_index,
                vectors,
                m=self.m,
                ef_construction=self.upper_level_ef_construction
                or self.ef_construction,
                max_share=self.upper_level_max_share,
                protect_last_in_edge=self.upper_level_protect_last_in_edge,
                skip_if_reachable=self.upper_level_skip_if_reachable,
            )
        except Exception as e:
            logger.warning(
                f"Upper-level projection failed, keeping the plain CAGRA graph: {e}",
                exc_info=True,
            )

    def _do_convert_gpu_to_cpu_index(
        self,
        faiss_gpu_build_index_output: FaissGpuBuildIndexOutput,
        vectors: Optional[np.ndarray] = None,
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
            if self.skip_stored_vectors:
                logger.debug(
                    "skip_stored_vectors=True: skipping vector storage during copyTo"
                )
                faiss_gpu_build_index_output.gpu_index.copyTo(cpu_index, True)
            else:
                faiss_gpu_build_index_output.gpu_index.copyTo(cpu_index)

            if self.project_link_to_base_layer and self.base_level_only:
                self._project_link_to_base_layer(cpu_index, vectors)

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
        self,
        faiss_gpu_build_index_output: FaissGpuBuildIndexOutput,
        vectors: Optional[np.ndarray] = None,
    ) -> FaissCpuBuildIndexOutput:
        """
        Method to convert a GPU Vector Search Index to CPU Index
        Returns a CPU read compatible vector search index
        Uses faiss specific library methods to achieve this.

        Args:
        faiss_gpu_build_index_output (FaissGpuBuildIndexOutput) A datamodel containing the GPU Faiss Index
        and dataset Vector Ids components
        vectors (np.ndarray, optional): The host array the GPU index was built from, in id order.
        Required when project_link_to_base_layer is true; layer 1 is built from it.

        Returns:
        FaissCpuBuildIndexOutput: A datamodel containing the created CPU Faiss Index
        and dataset Vector Ids components
        """

        if self.vector_dtype != DataType.BINARY:
            return self._do_convert_gpu_to_cpu_index(
                faiss_gpu_build_index_output, vectors
            )

        return self._do_convert_gpu_to_cpu_binary_index(faiss_gpu_build_index_output)
