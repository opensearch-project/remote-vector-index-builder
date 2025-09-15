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
from core.common.models.index_build_parameters import DataType
from core.common.models.index_builder import CagraGraphBuildAlgo
from core.index_builder.interface import IndexBuildService
from timeit import default_timer as timer

import logging

logger = logging.getLogger(__name__)


class FaissIndexBuildService(IndexBuildService):
    """
    Class exposing the build_gpu_index method for building a CPU read compatible Faiis GPU Index
    """

    def __init__(self):
        self.omp_num_threads = get_omp_num_threads()
        self.PQ_DIM_COMPRESSION_FACTOR = 4

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
        faiss_gpu_build_index_output = None
        faiss_cpu_build_index_output = None

        try:
            # Set number of threads for parallel processing
            faiss.omp_set_num_threads(self.omp_num_threads)

            # Step 1a: Create a structured GPUIndexConfig having defaults,
            # from a partial dictionary set from index build params
            if index_build_parameters.data_type != DataType.BINARY:
                gpu_index_config_params = {
                    "ivf_pq_params": {
                        "n_lists": calculate_ivf_pq_n_lists(
                            index_build_parameters.doc_count
                        ),
                        "pq_dim": int(
                            index_build_parameters.dimension
                            / self.PQ_DIM_COMPRESSION_FACTOR
                        ),
                    },
                    "graph_degree": index_build_parameters.index_parameters.algorithm_parameters.m
                    * 2,
                    "intermediate_graph_degree": index_build_parameters.index_parameters.algorithm_parameters.m
                    * 4,
                }
            else:
                gpu_index_config_params = {
                    "graph_build_algo": CagraGraphBuildAlgo.NN_DESCENT,
                    "graph_degree": index_build_parameters.index_parameters.algorithm_parameters.m
                    * 2,
                    "intermediate_graph_degree": index_build_parameters.index_parameters.algorithm_parameters.m
                    * 4,
                }

            faiss_gpu_index_cagra_builder = FaissGPUIndexCagraBuilder.from_dict(
                gpu_index_config_params
            )

            # Step 1b: create a GPU Index from the faiss config and vector dataset
            t1 = timer()
            faiss_gpu_build_index_output = (
                faiss_gpu_index_cagra_builder.build_gpu_index(
                    vectors_dataset,
                    index_build_parameters.dimension,
                    index_build_parameters.index_parameters.space_type,
                )
            )
            t2 = timer()
            index_build_time = t2 - t1
            logger.debug(
                f"Index build time for vector path {index_build_parameters.vector_path}: "
                f"{index_build_time:.2f} seconds"
            )

            # Step 2a: Create a structured CPUIndexConfig having defaults,
            # from a partial dictionary set from index build params
            cpu_index_config_params = {
                "ef_search": index_build_parameters.index_parameters.algorithm_parameters.ef_search,
                "ef_construction": index_build_parameters.index_parameters.algorithm_parameters.ef_construction,
                "vector_dtype": index_build_parameters.data_type,
            }
            faiss_index_hnsw_cagra_builder = FaissIndexHNSWCagraBuilder.from_dict(
                cpu_index_config_params
            )

            # Step 2b: Convert GPU Index to CPU Index, update index to cpu index in index-id mappings
            # Also Delete GPU Index after conversion

            t1 = timer()
            faiss_cpu_build_index_output = (
                faiss_index_hnsw_cagra_builder.convert_gpu_to_cpu_index(
                    faiss_gpu_build_index_output
                )
            )
            t2 = timer()
            index_conversion_time = t2 - t1
            logger.debug(
                f"Index conversion time for vector path {index_build_parameters.vector_path}: "
                f"{index_conversion_time:.2f} seconds"
            )

            # Step 3: Write CPU Index to persistent storage
            t1 = timer()
            faiss_index_hnsw_cagra_builder.write_cpu_index(
                faiss_cpu_build_index_output, cpu_index_output_file_path
            )
            t2 = timer()
            index_write_time = t2 - t1
            logger.debug(
                f"Index write time for vector path {index_build_parameters.vector_path}: "
                f"{index_write_time:.2f} seconds"
            )

        except Exception as exception:
            # Clean up GPU Index Response if orchestrator failed after GPU Index Creation
            if faiss_gpu_build_index_output is not None:
                try:
                    faiss_gpu_build_index_output.cleanup()
                except Exception as e:
                    logger.error(
                        f"Failed to clean up GPU index response for vector path "
                        f"{index_build_parameters.vector_path}: {e}"
                    )

            # Clean up CPU Index Response if orchestrator failed after CPU Index Creation
            if faiss_cpu_build_index_output is not None:
                try:
                    faiss_cpu_build_index_output.cleanup()
                except Exception as e:
                    logger.error(
                        f"Failed to clean up CPU index response for vector path "
                        f"{index_build_parameters.vector_path}: {e}"
                    )

            raise Exception(
                f"Faiss Index Build Service build_index workflow failed: {exception}"
            ) from exception
