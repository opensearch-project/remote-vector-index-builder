# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
from core.common.models import (
    IndexBuildParameters,
    VectorsDataset,
)

from core.common.models.index_builder.faiss import (
    FaissGPUIndexCagraBuilder,
    FaissIndexHNSWCagraBuilder,
)
from core.index_builder.index_builder_utils import (
    calculate_ivf_pq_n_lists,
    get_omp_num_threads,
)
from core.index_builder.interface import IndexBuildService


class FaissIndexBuildService(IndexBuildService):
    """
    Class exposing the build_gpu_index method for building a CPU read compatible Faiis GPU Index
    """

    def __init__(self):
        self.omp_num_threads = get_omp_num_threads()

    def build_index(
        self,
        index_build_parameters: IndexBuildParameters,
        vectors_dataset: VectorsDataset,
        cpu_index_output_file_path: str,
    ) -> None:
        """
        Orchestrates the workflow of
        - creating a GPU Index for the specified vectors dataset,
        - converting into CPU compatible Index
        - and writing the CPU Index to disc
        Uses the faiss library methods to achieve this.

        Args:
            vectors_dataset: The set of vectors to index
            index_build_parameters: The API Index Build parameters
            cpu_index_output_file_path: The complete file path on disc to write the cpuIndex to.
        """
        faiss_gpu_index_cagra_builder = None
        faiss_index_hnsw_cagra_builder = None
        gpu_faiss_index_id_map = None

        try:
            # Set number of threads for parallel processing
            faiss.omp_set_num_threads(self.omp_num_threads)

            # Step 1a: Create a structured GPUIndexConfig having defaults,
            # from a partial dictionary set from index build params
            gpu_index_config_params = {
                "ivf_pq_build_params": {
                    "n_lists": calculate_ivf_pq_n_lists(
                        index_build_parameters.doc_count
                    ),
                    "pq_dim": index_build_parameters.dimension,
                }
            }
            faiss_gpu_index_cagra_builder = FaissGPUIndexCagraBuilder.from_dict(
                gpu_index_config_params
            )

            # Step 1b: create a GPU Index from the faiss config and vector dataset
            gpu_faiss_index_id_map = faiss_gpu_index_cagra_builder.build_gpu_index(
                vectors_dataset,
                index_build_parameters.dimension,
                index_build_parameters.index_parameters.space_type,
            )

            # Step 2a: Create a structured CPUIndexConfig having defaults,
            # from a partial dictionary set from index build params
            cpu_index_config_params = {
                "ef_search": index_build_parameters.index_parameters.algorithm_parameters.ef_search,
                "ef_construction": index_build_parameters.index_parameters.algorithm_parameters.ef_construction,
            }
            faiss_index_hnsw_cagra_builder = FaissIndexHNSWCagraBuilder.from_dict(
                cpu_index_config_params
            )

            # Step 2b: Convert GPU Index to CPU Index, update index to cpu index in index-id mappings
            # Also Delete GPU Index after conversion
            cpu_faiss_index_id_map = (
                faiss_index_hnsw_cagra_builder.convert_gpu_to_cpu_index(
                    gpu_faiss_index_id_map
                )
            )

            # Step 3: Write CPU Index to persistent storage
            faiss_index_hnsw_cagra_builder.write_cpu_index(
                cpu_faiss_index_id_map, cpu_index_output_file_path
            )

        except Exception as exception:
            # Clean up GPU Index Response if orchestrator failed after GPU Index Creation
            if gpu_faiss_index_id_map is not None:
                try:
                    del gpu_faiss_index_id_map
                except Exception as e:
                    print(f"Warning: Failed to clean up GPU index response: {str(e)}")

            # Clean up CPU Index Response if orchestrator failed after CPU Index Creation
            if cpu_faiss_index_id_map is not None:
                try:
                    del cpu_faiss_index_id_map
                except Exception as e:
                    print(f"Warning: Failed to clean up CPU index response: {str(e)}")
            raise Exception(
                f"Faiss Index Build Service build_index workflow failed. Reason: {str(exception)}"
            ) from exception
