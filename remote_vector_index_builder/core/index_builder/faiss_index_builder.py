# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from core.common.models import (
    GPUIndexBuildConfig,
    GPUIndexCagraConfig,
    IndexHNSWCagraConfig,
    SpaceType,
)
from core.index_builder.faiss_index_config_builder import (
    FaissIndexConfigBuilder,
)
import faiss
from core.common.models.index_build_parameters import (
    IndexBuildParameters,
)
from core.common.models.vectors_dataset import (
    VectorsDataset,
)
from core.index_builder.index_config_builder import (
    IndexConfigBuilder,
)
from core.index_builder.index_config_director import (
    IndexConfigDirector,
)
from core.index_builder.models.create_gpu_index_response import (
    CreateGPUIndexResponse,
)
from core.index_builder.index_builder_utils import (
    calculate_ivf_pq_n_lists,
    configure_metric,
    get_omp_num_threads,
)


class FaissIndexBuilder:
    """
    Class exposing the the build_gpu_index method for building a CPU read compatible Faiis GPU Index
    """

    def __init__(self):
        self.gpu_resources = faiss.StandardGpuResources()
        self.omp_num_threads = get_omp_num_threads()

    def _create_gpu_index(
        self,
        vectorsDataset: VectorsDataset,
        dataset_dimension: int,
        faissGPUIndexCagraConfig: faiss.GpuIndexCagraConfig,
        space_type: SpaceType = SpaceType.L2,
    ):
        # Configure the distance metric
        metric = configure_metric(space_type)

        # Create GPU CAGRA index with specified configuration
        faiss_gpu_index_cagra = faiss.GpuIndexCagra(
            self.gpu_resources, dataset_dimension, metric, faissGPUIndexCagraConfig
        )

        # Create ID mapping layer to preserve document IDs
        faiss_id_map_index = faiss.IndexIDMap(faiss_gpu_index_cagra)
        # Add vectors and their corresponding IDs to the index
        faiss_id_map_index.add_with_ids(vectorsDataset.vectors, vectorsDataset.doc_ids)

        # Clean up memory by releasing vector dataset
        vectorsDataset.free_vectors_space()
        return CreateGPUIndexResponse(
            gpu_index_cagra=faiss_gpu_index_cagra, id_map_index=faiss_id_map_index
        )

    def _create_and_write_cpu_index_to_file(
        self,
        create_gpu_index_response: CreateGPUIndexResponse,
        index_hnsw_cagra_config: IndexHNSWCagraConfig,
        cpuIndexOutputFilePath: str,
    ):
        """
        Method to Create and Write the CPU compatible Index from a GPU Index

        Args:
        create_gpu_index_response (CreateGPUIndexResponse): datamodel containing the GPU Index and Dataset ID Maps
        index_hnsw_cagra_config (IndexHNSWCagraConfig): CPU Search Index config
        cpuIndexFileOutputpath: Complete File path to write the CPU Index to
        """

        # Initialize CPU Index
        cpuIndex = faiss.IndexHNSWCagra()

        # Configure HNSW Search parameters
        cpuIndex.hnsw.efConstruction = index_hnsw_cagra_config.ef_construction
        cpuIndex.hnsw.efSearch = index_hnsw_cagra_config.ef_search
        cpuIndex.base_level_only = index_hnsw_cagra_config.base_level_only
        cpuIndex.own_fields = index_hnsw_cagra_config.own_fields

        # Copy GPU index to CPU index
        create_gpu_index_response.gpu_index_cagra.copyTo(cpuIndex)

        # Update the ID map index with the CPU index
        create_gpu_index_response.id_map_index.index = cpuIndex

        # TODO: Investigate what issues may arise while writing index to local file
        # Write the final index to disk
        faiss.write_index(
            create_gpu_index_response.id_map_index, cpuIndexOutputFilePath
        )

        # Clean up CPU index after writing to file
        del cpuIndex

    def _create_gpu_index_build_config(self, **kwargs) -> GPUIndexBuildConfig:
        """
        Create an index configuration using the builder pattern.

        Args:
            **kwargs: Configuration parameters including 'hnsw_config', 'gpu_config', and 'metric'.

        Returns:
            GPUIndexBuildConfig: The constructed index configuration.

        Raises:
            ValueError: If required configuration parameters are missing.
        """
        builder = IndexConfigBuilder()
        director = IndexConfigDirector(builder)
        return director.construct_config(kwargs)

    def _create_faiss_gpu_index_config(self, config: GPUIndexCagraConfig):
        """
        Create an faiss index configuration using the builder pattern.

        Args:
            config: GPUIndexCagraConfig: The core datamodel in remote_vector_index_builder

        Returns:
            faiss.GpuIndexCagraConfig: The equivalent faiss config
        """
        faissIndexBuilder = FaissIndexConfigBuilder()
        return faissIndexBuilder.with_gpu_config(config).build_gpu_index_cagra_config()

    def build_gpu_index(
        self,
        vectorsDataset: VectorsDataset,
        indexBuildParameters: IndexBuildParameters,
        cpuIndexOutputFilePath: str,
    ):
        """
        Creates a GPU Index for the specified vectors dataset, coonverts into CPU compatible Index
        and writes the CPU Index to disc

        Args:
            vectorsDataset: The set of vectors to index
            indexBuildParameters: The API Index Build parameters
            cpuIndexOutputFilePath: The complete file path on disc to write the cpuIndex to.
        """
        # Set number of threads for parallel processing
        faiss.omp_set_num_threads(self.omp_num_threads)

        # Create a structured GPUIndexBuildConfig having defaults, from a partial dictionary set with index params
        index_build_config = self._create_gpu_index_build_config(
            hnsw_config={},
            gpu_config={
                "ivf_pq_build_params": {
                    "pq_dim": indexBuildParameters.dimension,
                    "n_lists": calculate_ivf_pq_n_lists(indexBuildParameters.doc_count),
                }
            },
            metric=indexBuildParameters.index_parameters.space_type,
        )

        # Create a faiis equivalent version of gpu index build config
        faiss_gpu_index_cagra_config = self._create_faiss_gpu_index_config(
            index_build_config.gpu_index_cagra_config
        )

        index_hnsw_cagra_config = index_build_config.index_hnsw_cagra_config

        # create a GPU Index from the faiss config and vector dataset
        create_gpu_index_response = self._create_gpu_index(
            vectorsDataset,
            indexBuildParameters.dimension,
            faiss_gpu_index_cagra_config,
            index_build_config.metric,
        )

        # Convert the GPU Index to CPU Index and write to disk
        self._create_and_write_cpu_index_to_file(
            create_gpu_index_response, index_hnsw_cagra_config, cpuIndexOutputFilePath
        )

        # Clean up GPU index response
        del create_gpu_index_response
