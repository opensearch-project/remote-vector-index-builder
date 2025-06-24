# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
import pytest

from core.common.models import SpaceType

from core.common.models.index_builder import (
    CagraGraphBuildAlgo,
    FaissGpuBuildIndexOutput,
)

from core.common.models.index_builder.faiss import (
    IVFPQBuildCagraConfig,
    IVFPQSearchCagraConfig,
    FaissGPUIndexCagraBuilder,
)
from unittest.mock import Mock
import gc


class TestFaissGPUIndexCagraBuilder:

    @pytest.fixture
    def default_builder(self):
        return FaissGPUIndexCagraBuilder()

    @pytest.fixture
    def custom_params(self):
        params = {
            "intermediate_graph_degree": 128,
            "graph_degree": 64,
            "store_dataset": True,
            "refine_rate": 3.0,
            "graph_build_algo": CagraGraphBuildAlgo.IVF_PQ,
            "ivf_pq_params": {"n_lists": 2048},
            "ivf_pq_search_params": {"n_probes": 16},
        }
        return params

    @pytest.fixture
    def custom_builder(self, custom_params):
        builder = FaissGPUIndexCagraBuilder.from_dict(custom_params)
        builder.device = 1  # Set device after initialization
        return builder

    def test_default_initialization(self, default_builder):
        assert default_builder.intermediate_graph_degree == 64
        assert default_builder.graph_degree == 32
        assert default_builder.graph_build_algo == CagraGraphBuildAlgo.IVF_PQ
        assert default_builder.store_dataset is False
        assert default_builder.refine_rate == 1.0
        assert isinstance(default_builder.ivf_pq_build_config, IVFPQBuildCagraConfig)
        assert isinstance(default_builder.ivf_pq_search_config, IVFPQSearchCagraConfig)

    def test_custom_initialization(self, custom_builder):
        assert custom_builder.intermediate_graph_degree == 128
        assert custom_builder.graph_degree == 64
        assert custom_builder.store_dataset is True
        assert custom_builder.refine_rate == 3.0
        assert custom_builder.graph_build_algo == CagraGraphBuildAlgo.IVF_PQ
        assert custom_builder.device == 1

    def test_configure_build_algo(self, default_builder):
        algo = default_builder._configure_build_algo()
        assert algo == faiss.graph_build_algo_IVF_PQ

    @pytest.mark.parametrize(
        "params,error_msg",
        [
            (
                {"intermediate_graph_degree": 0},
                "intermediate_graph_degree must be positive",
            ),
            (
                {"intermediate_graph_degree": -1},
                "intermediate_graph_degree must be positive",
            ),
            ({"graph_degree": 0}, "graph_degree must be positive"),
            ({"graph_degree": -1}, "graph_degree must be positive"),
            ({"device": -1}, "device must be non-negative"),
        ],
    )
    def test_validate_params_invalid(self, params, error_msg):
        with pytest.raises(
            ValueError, match=f"FaissGPUIndexCagraBuilder param: {error_msg}"
        ):
            FaissGPUIndexCagraBuilder._validate_params(params)

    def test_to_faiss_config(self, custom_builder):
        config = custom_builder.to_faiss_config()

        assert isinstance(config, faiss.GpuIndexCagraConfig)
        assert config.intermediate_graph_degree == 128
        assert config.graph_degree == 64
        assert config.store_dataset is True
        assert config.device == 1
        assert config.refine_rate == 3.0
        assert config.build_algo == faiss.graph_build_algo_IVF_PQ

        # Test IVF-PQ configurations if present
        if custom_builder.graph_build_algo == CagraGraphBuildAlgo.IVF_PQ:
            assert config.ivf_pq_params is not None
            assert config.ivf_pq_search_params is not None

    def test_from_dict_custom(self, custom_params):

        builder = FaissGPUIndexCagraBuilder.from_dict(custom_params)
        assert isinstance(builder, FaissGPUIndexCagraBuilder)

        assert (
            builder.intermediate_graph_degree
            == custom_params["intermediate_graph_degree"]
        )
        assert builder.graph_degree == custom_params["graph_degree"]
        assert builder.store_dataset == custom_params["store_dataset"]
        assert builder.refine_rate == custom_params["refine_rate"]
        assert builder.graph_build_algo == custom_params["graph_build_algo"]

        assert (
            builder.ivf_pq_build_config.n_lists
            == custom_params["ivf_pq_params"]["n_lists"]
        )
        assert (
            builder.ivf_pq_search_config.n_probes
            == custom_params["ivf_pq_search_params"]["n_probes"]
        )

    def test_build_gpu_index_success(self, default_builder, vectors_dataset):
        """Test successful GPU index building"""
        # Configure mock returns

        # Execute build
        result = default_builder.build_gpu_index(
            vectors_dataset, dataset_dimension=3, space_type=SpaceType.INNERPRODUCT
        )

        # Verify result
        assert isinstance(result, FaissGpuBuildIndexOutput)
        assert isinstance(result.gpu_index, faiss.GpuIndexCagra)
        assert isinstance(result.index_id_map, faiss.IndexIDMap)
        assert not result.index_id_map.is_deleted
        assert not result.gpu_index.is_deleted

    def test_build_gpu_index_config_error(self, default_builder, vectors_dataset):
        default_builder.to_faiss_config = Mock(side_effect=Exception("Config error"))

        with pytest.raises(Exception) as exc_info:
            default_builder.build_gpu_index(
                vectors_dataset, dataset_dimension=3, space_type=SpaceType.L2
            )

        assert "Failed to create faiss GPU index config" in str(exc_info.value)

    def test_build_gpu_index_cleanup_on_error(
        self, default_builder, vectors_dataset, deletion_tracker
    ):
        """Test cleanup when error occurs during index building"""
        # Create GPU index that will be cleaned up
        gpu_index = faiss.GpuIndexCagra()
        gpu_index_id = gpu_index.id
        original_index_id_map = faiss.IndexIDMap

        try:
            # Make IndexIDMap raise an error
            faiss.IndexIDMap = Mock(side_effect=Exception("Index error"))

            with pytest.raises(Exception):
                default_builder.build_gpu_index(
                    vectors_dataset, dataset_dimension=3, space_type=SpaceType.L2
                )

            # Force garbage collection to ensure __del__ is called
            gpu_index = None
            gc.collect()

            # Verify GPU index was cleaned up
            assert deletion_tracker.is_deleted(gpu_index_id)

        finally:
            # Restore original IndexIDMap
            faiss.IndexIDMap = original_index_id_map

    def test_build_gpu_index_resource_cleanup(
        self, default_builder, vectors_dataset, deletion_tracker
    ):
        """Test resource cleanup during normal operation"""
        result = default_builder.build_gpu_index(
            vectors_dataset, dataset_dimension=3, space_type=SpaceType.L2
        )

        # Store IDs before cleanup
        gpu_index_id = result.gpu_index.id
        index_id_map_id = result.index_id_map.id

        # Verify initial state
        assert not deletion_tracker.is_deleted(gpu_index_id)
        assert not deletion_tracker.is_deleted(index_id_map_id)

        # Clean up
        del result
        gc.collect()

        # Verify cleanup
        assert deletion_tracker.is_deleted(gpu_index_id)
        assert deletion_tracker.is_deleted(index_id_map_id)
