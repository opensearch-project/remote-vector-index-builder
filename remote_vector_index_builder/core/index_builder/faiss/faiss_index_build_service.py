# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
from typing import cast
from core.common.models import (
    GPUIndexConfig,
    SpaceType,
    IndexBuildParameters,
    VectorsDataset,
    GPUIndexResponse,
    CPUIndexResponse,
    CPUIndexConfig,
)

from core.common.models.index_builder.faiss import (
    FaissGPUIndexResponse,
    FaissCPUIndexResponse,
    GPUIndexCagraConfig,
    IndexHNSWCagraConfig,
)
from core.index_builder.index_builder_utils import (
    calculate_ivf_pq_n_lists,
    configure_metric,
    get_omp_num_threads,
)
from core.index_builder.interface import (
    IndexBuildService,
    GPUIndexBuildService,
    GPUToCPUIndexConverter,
    CPUIndexWriter,
)


class FaissIndexBuildService(
    IndexBuildService, GPUIndexBuildService, GPUToCPUIndexConverter, CPUIndexWriter
):
    """
    Class exposing the build_gpu_index method for building a CPU read compatible Faiis GPU Index
    """

    def __init__(self):
        self.gpu_resources = faiss.StandardGpuResources()
        self.omp_num_threads = get_omp_num_threads()

    def build_gpu_index(
        self,
        vectorsDataset: VectorsDataset,
        dataset_dimension: int,
        gpu_index_config: GPUIndexConfig,
        space_type: SpaceType,
    ):
        """
        Method to create a GPU Index using the provided configuration

        Args:
        vectorsDataset (VectorsDataset): VectorsDataset object containing vectors and document IDs
        dataset_dimension (int): Dimension of the vectors
        faissGPUIndexCagraConfig (faiss.GpuIndexCagraConfig): GPU Index configuration
        space_type (SpaceType, optional): Distance metric to be used (defaults to L2)

        Returns:
        CreateGPUIndexResponse: A data model containing the created GPU Index and ID Map Index
        """
        faiss_gpu_index = None
        faiss_id_map_index = None
        faiss_gpu_index_config = None

        # Create a faiis equivalent version of gpu index build config
        try:
            faiss_gpu_index_config = cast(
                GPUIndexCagraConfig, gpu_index_config
            ).to_faiss_config()
        except Exception as e:
            raise Exception(f"Failed to create faiss GPU index config: {str(e)}") from e

        try:
            # Configure the distance metric
            metric = configure_metric(space_type)

            # Create GPU CAGRA index with specified configuration
            faiss_gpu_index = faiss.GpuIndexCagra(
                self.gpu_resources, dataset_dimension, metric, faiss_gpu_index_config
            )

            # Create ID mapping layer to preserve document IDs
            faiss_id_map_index = faiss.IndexIDMap(faiss_gpu_index)
            # Add vectors and their corresponding IDs to the index
            faiss_id_map_index.add_with_ids(
                vectorsDataset.vectors, vectorsDataset.doc_ids
            )

            return FaissGPUIndexResponse(
                gpu_index_cagra=faiss_gpu_index, index_id_map=faiss_id_map_index
            )
        except Exception as e:
            if faiss_gpu_index is not None:
                del faiss_gpu_index
            if faiss_id_map_index is not None:
                del faiss_id_map_index
            raise Exception(f"Failed to create faiss GPU index: {str(e)}") from e

    def convert_gpu_to_cpu_index(
        self,
        gpu_index_response: GPUIndexResponse,
        cpu_index_config: CPUIndexConfig,
    ) -> CPUIndexResponse:
        """
        Method to convert a GPU Vector Search Index to CPU Index
        Returns a CPU read compatible vector search index
        Uses faiss specific library methods to achieve this.

        Args:
        gpu_index_response (GPUIndexResponse) FaissGPUIndexResponse with gpu index and gpu index-vectors id map.
        cpu_index_config (CPUIndexConfig): IndexHNSWCagraConfig containing the IndexHNSWCagra CPU Index params

        Returns:
        cpu_index_response (CPUIndexResponse) FaissCPUIndexResponse with cpu index and cpu index-vectors id map.
        """
        cpuIndex = None
        try:
            cpu_index_config = cast(IndexHNSWCagraConfig, cpu_index_config)
            gpu_index_response = cast(FaissGPUIndexResponse, gpu_index_response)
            # Initialize CPU Index
            cpuIndex = faiss.IndexHNSWCagra()

            # Configure CPU Index parameters
            cpuIndex.hnsw.efConstruction = cpu_index_config.ef_construction
            cpuIndex.hnsw.efSearch = cpu_index_config.ef_search
            cpuIndex.base_level_only = cpu_index_config.base_level_only

            # Copy GPU index to CPU index
            gpu_index_response.gpu_index_cagra.copyTo(cpuIndex)

            # Update the ID map index with the CPU index
            gpu_index_response.index_id_map.index = cpuIndex

            return FaissCPUIndexResponse(
                cpu_index=cpuIndex, index_id_map=gpu_index_response.index_id_map
            )
        except Exception as e:
            raise Exception(
                f"Failed to convert GPU index to CPU index: {str(e)}"
            ) from e

    def write_cpu_index(
        self, cpu_index_response: CPUIndexResponse, cpu_index_output_file_path: str
    ) -> None:
        """
        Method to write the CPU index and vector dataset id mapping to persistent local file path
        for uploading later to remote object store.
        Uses faiss write_index library method to achieve this

        Args:
        cpu_index_response (CPUIndexResponse): response model containing the CPU Index and Index-Vector IDs Map
        cpu_index_output_file_path (str): File path to persist Index-Vector IDs map to
        """
        try:
            cpu_index_response = cast(FaissCPUIndexResponse, cpu_index_response)

            # TODO: Investigate what issues may arise while writing index to local file
            # Write the final cpu index - vectors id mapping to disk
            faiss.write_index(
                cpu_index_response.index_id_map, cpu_index_output_file_path
            )
        except IOError as io_error:
            raise Exception(
                f"Failed to write index to file {cpu_index_output_file_path}: {str(io_error)}"
            ) from io_error
        except Exception as e:
            raise Exception(
                f"Unexpected error while writing index to file: {str(e)}"
            ) from e

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
        gpu_index_config = None
        cpu_index_config = None
        gpu_index_response = None

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
            gpu_index_config = GPUIndexCagraConfig.from_dict(gpu_index_config_params)

            # Step 1b: create a GPU Index from the faiss config and vector dataset
            gpu_index_response = self.build_gpu_index(
                vectors_dataset,
                index_build_parameters.dimension,
                gpu_index_config,
                index_build_parameters.index_parameters.space_type,
            )

            # Step 2a: Create a structured CPUIndexConfig having defaults,
            # from a partial dictionary set from index build params
            cpu_index_config_params = {
                "ef_search": index_build_parameters.index_parameters.algorithm_parameters.ef_search,
                "ef_construction": index_build_parameters.index_parameters.algorithm_parameters.ef_construction,
            }
            cpu_index_config = IndexHNSWCagraConfig.from_dict(cpu_index_config_params)

            # Step 2b: Convert GPU Index to CPU Index, update index to cpu index in index-id mappings
            cpu_index_response = self.convert_gpu_to_cpu_index(
                gpu_index_response, cpu_index_config
            )

            # Step 3: Write CPU Index to persistent storage
            self.write_cpu_index(cpu_index_response, cpu_index_output_file_path)

        except Exception as e:
            raise Exception(
                f"Faiss Orchestrater build_index workflow failed. Reason: {str(e)}"
            ) from e
        finally:
            # Clean up created indexes
            if gpu_index_response is not None:
                try:
                    del gpu_index_response
                except Exception as e:
                    print(f"Warning: Failed to clean up GPU index response: {str(e)}")
            if cpu_index_response is not None:
                try:
                    del cpu_index_response
                except Exception as e:
                    print(f"Warning: Failed to clean up CPU index response: {str(e)}")
